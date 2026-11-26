"""Check the fixed PatchTST artifact and its unchanged state after revision.

Run only in an isolated candidate execution sandbox, never in the private scorer.
Matching artifacts do not prove their training provenance or actual inference use.
"""
import argparse,json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
CONTRACT=json.loads((HERE/'backbone_contract.json').read_text())

def inside(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('backbone artifact must be a file inside the model directory')
    return p

def check_states(before,after,horizon):
    expected=CONTRACT['state_layout'][str(horizon)]
    for label,state in [('pretrained',before),('after_revision',after)]:
        if set(state)!=set(expected):raise ValueError(label+' backbone keys differ from fixed PatchTST')
        for key,spec in expected.items():
            value=state[key]
            if not isinstance(value,torch.Tensor) or list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype'] or not torch.isfinite(value).all():
                raise ValueError(label+' backbone shape/dtype/value mismatch: '+key)
    for key,value in before.items():
        if not torch.equal(value,after[key]):raise ValueError('revision changed backbone parameter or buffer: '+key)
    return len(expected)

def check_horizon(model_dir,horizon):
    root=Path(model_dir)
    manifest=root/'research_backbone.json'
    if manifest.exists():
        entry=json.loads(manifest.read_text())['horizons'][str(horizon)]
        before_path=inside(root,entry['pretrained_state'])
        after_path=inside(root,entry['final_state'])
        config_path=inside(root,entry['pretrained_config'])
        prefix=entry.get('final_prefix','')
    else:
        bases=sorted((root/f'h{horizon}').glob('*_PatchTST_*'))
        finals=sorted((root/f'h{horizon}').glob('*_PIR_*'))
        if len(bases)!=1 or len(finals)!=1:raise ValueError('expected one pretrained and one revision model per horizon')
        before_path=inside(root,str((bases[0]/'checkpoint.pth').relative_to(root)))
        after_path=inside(root,str((finals[0]/'checkpoint.pth').relative_to(root)))
        config_path=inside(root,str((bases[0]/'config.json').relative_to(root)))
        prefix='model.'
    cfg=json.loads(config_path.read_text())
    for key,value in {**CONTRACT['fixed_architecture'],**CONTRACT['pretraining_recipe'],'pred_len':horizon}.items():
        if cfg.get(key)!=value:raise ValueError('pretrained configuration: '+key)
    before=torch.load(before_path,map_location='cpu',weights_only=True)
    full=torch.load(after_path,map_location='cpu',weights_only=True)
    after={k[len(prefix):]:v for k,v in full.items() if k.startswith(prefix)} if prefix else full
    count=check_states(before,after,horizon)
    return {'horizon':horizon,'backbone_tensors_checked':count,'weights_and_buffers_unchanged':True}

def check(model_dir,horizons=(96,192,336,720)):
    return {'pass':True,'horizons':[check_horizon(model_dir,h) for h in horizons],
            'audit_required':['authors pretraining initialization/data/recipe were actually used','the checked backbone implements the fixed PatchTST forward and supplies the forecasts being revised','revision history contains training rows only']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--horizon',type=int,choices=[96,192,336,720]);a=p.parse_args()
    try:result=check(a.model,[a.horizon] if a.horizon else [96,192,336,720])
    except Exception as e:result={'pass':False,'error':str(e)}
    print(json.dumps(result));raise SystemExit(0 if result['pass'] else 1)

"""Check exported FedSPA state, protocol certificate, and independent predictions.
Run in the isolated candidate execution environment. Training-history declarations still require patch audit.
"""
import argparse,importlib.util,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from format import read_predictions

def check_training_ids(train_ids, data):
    with np.load(Path(data)/'train/train.npz',allow_pickle=False) as z:
        permitted=set(z['id'].tolist())
    if not train_ids or len(set(train_ids))!=len(train_ids) or not set(train_ids)<=permitted:
        raise ValueError('training IDs must be unique designated training rows; validation fitting is forbidden')

def check(model,input_path,pred_path,code=None,data=None):
    model=Path(model);card=json.loads((model/'model_card.json').read_text())
    fixed={'format':'fedspa-feature-card/v1','encoder':'CLIP ViT-B/16','encoder_frozen':True,
           'classes':101,'dimension':512,'shots_pool':8,'clients':10,'rho':.1,'rounds':10,'state':'model.npz'}
    for k,v in fixed.items():
        if card.get(k)!=v:raise ValueError('protocol certificate: '+k)
    uploads=card['uploads']
    if len(uploads)!=100:raise ValueError('upload count')
    for c in range(10):
        u=[r for r in uploads if r['client']==c]
        if len(u)!=10 or any(r['floats']>101*512 or r['floats']<0 or r['floats']!=np.prod(r['shape']) for r in u):raise ValueError('upload budget')
    data=Path(data or os.environ.get('BENCH_DATA_DIR',Path(__file__).resolve().parents[1]/'data'))
    check_training_ids(card['train_ids'],data)
    ids,p=read_predictions(pred_path)
    with np.load(input_path,allow_pickle=False) as z:
        if not np.array_equal(z['id'],ids):raise ValueError('IDs')
        x=torch.from_numpy(z['x'])
    torch.set_num_threads(max(1,min(4,int(os.environ.get('BENCH_CPUS','4')))))
    code=Path(code or os.environ.get('BENCH_CODE_DIR','.'))
    if (code/'model_api.py').is_file():
        # Candidate modules are imported only inside the isolated execution probe.
        sys.path.insert(0,str(code))
        spec=importlib.util.spec_from_file_location('candidate_api',code/'model_api.py')
        api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
        for idx in (np.arange(len(x)),np.arange(len(x)-1,-1,-1)[::2]):
            got=np.asarray(api.predict(model,x.numpy()[idx].copy()))
            if got.shape!=(len(idx),11) or not np.isfinite(got).all() or not np.array_equal(got,p[idx]):
                raise ValueError('generic model replay/subset mismatch')
        return {'pass':True,'interface':'generic','predictions_equal':True,'uploads':100,'max_upload_floats':max(r['floats'] for r in uploads),'audit_required':['actual communications','training data provenance','rounds','fitted state provenance']}
    if not np.isfinite([card['alpha'],card['beta']]).all():raise ValueError('nonfinite coefficients')
    with np.load(model/'model.npz',allow_pickle=False) as z:
        expected={'global','text'}|{f'{s}_{c}' for s in ['adapter','labels','semantic'] for c in range(10)}
        if set(z.files)!=expected:raise ValueError('state keys')
        for name in expected:
            shape=(101,101) if name.startswith('labels_') else (101,512)
            if z[name].shape!=shape or not np.isfinite(z[name]).all() or z[name].dtype!=np.float32:raise ValueError('state shape/dtype/finite: '+name)
        with torch.no_grad():
            predictions=[(100.0*x@torch.from_numpy(z['global']).t()).topk(1,1,True,True)[1].flatten()]
            for c in range(10):
                affinity=x@torch.from_numpy(z[f'adapter_{c}']).t()
                cache=(-(card['beta']-card['beta']*affinity)).exp()@torch.from_numpy(z[f'labels_{c}'])
                logits=100.0*x@torch.from_numpy(z[f'semantic_{c}']).t()+cache*card['alpha']
                predictions.append(logits.topk(1,1,True,True)[1].flatten())
            expected=torch.stack(predictions,dim=1).numpy()
    if not np.array_equal(p,expected):raise ValueError('certificate prediction mismatch')
    return {'pass':True,'predictions_equal':True,'uploads':100,'max_upload_floats':max(r['floats'] for r in uploads),'audit_required':['actual communications','training data provenance','number of training rounds','encoder provenance']}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('model');a.add_argument('--input',required=True);a.add_argument('--pred',required=True);a.add_argument('--code');a.add_argument('--data');args=a.parse_args()
    try:r=check(args.model,args.input,args.pred,args.code,args.data)
    except Exception as e:r={'pass':False,'genacc':float('inf'),'error':str(e)}
    print(json.dumps(r));sys.exit(0 if r['pass'] else 1)

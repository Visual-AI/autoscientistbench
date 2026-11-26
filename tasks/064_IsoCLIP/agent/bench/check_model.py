"""Isolated fixed-encoder probe; candidate code is never imported by the scorer.

Tests actual module structure, weights/buffers, preprocessing and encoder outputs.
Source audit must still establish that retrieval uses this checked path and that
no external data or encoder adaptation was used elsewhere.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch

def module_structure(model):
    return [(name,type(module).__name__,module.extra_repr()) for name,module in model.named_modules()]

def same_frozen_encoder(candidate,reference):
    if module_structure(candidate)!=module_structure(reference):raise ValueError('fixed encoder module structure changed')
    a,b=candidate.state_dict(),reference.state_dict()
    if set(a)!=set(b):raise ValueError('fixed encoder state keys changed')
    for key,value in b.items():
        if a[key].dtype!=value.dtype or a[key].shape!=value.shape or not torch.equal(a[key],value):
            raise ValueError('fixed encoder parameter or buffer changed: '+key)
    if any(p.requires_grad for p in candidate.parameters()):raise ValueError('encoder parameters must be frozen')
    if any(m.training for m in candidate.modules()):raise ValueError('encoder must remain in evaluation mode')

def compare_output(candidate,reference,label):
    if candidate.shape!=reference.shape or candidate.dtype!=reference.dtype or not torch.isfinite(candidate).all() or not torch.allclose(candidate,reference,rtol=1e-5,atol=1e-6):
        raise ValueError(label+' does not match the frozen encoder')

@torch.no_grad()
def fixed_image_features(model,images):
    visual=model.visual
    x=visual.conv1(images).reshape(images.shape[0],visual.conv1.out_channels,-1).permute(0,2,1)
    x=torch.cat([visual.class_embedding.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device),x],dim=1)
    x=visual.ln_pre(x+visual.positional_embedding.to(x.dtype))
    x=visual.transformer(x.permute(1,0,2)).permute(1,0,2)
    return visual.ln_post(x[:,0,:])

def check(model_dir,checkpoint,code):
    import numpy as np
    from PIL import Image
    from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from reference_encoder import build_model
    # Only the fixed, operator-supplied checkpoint is loaded as TorchScript.
    state=torch.jit.load(str(checkpoint),map_location='cpu').state_dict()
    reference=build_model(state).float().eval().requires_grad_(False)
    if reference.visual.input_resolution!=224 or tuple(reference.visual.conv1.weight.shape)!=(768,3,32,32):
        raise ValueError('reference must be the supplied CLIP ViT-B/32')
    preprocess=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),lambda x:x.convert('RGB'),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    sys.path[:0]=[str(Path(code)/'vendor'),str(code)]
    spec=importlib.util.spec_from_file_location('candidate_encoder_api',Path(code)/'encoder_api.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    candidate,transform=api.load_encoder(Path(model_dir),Path(checkpoint))
    same_frozen_encoder(candidate,reference)
    image=Image.fromarray(np.random.default_rng(71).integers(0,256,(231,247,3),dtype=np.uint8))
    x=preprocess(image);got=transform(image);compare_output(got,x,'image preprocessing')
    with torch.no_grad():compare_output(api.encode(candidate,x[None]),fixed_image_features(reference,x[None]),'preprojection image features')
    same_frozen_encoder(candidate,reference)
    return {'pass':True,'structure_and_state_before_after':True,'preprocess_and_features':True,'audit_required':['retrieval actually uses this frozen encoder path','no other encoder or pretrained resource','projector/postprocessing provenance']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--checkpoint',required=True);p.add_argument('--code',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    try:result=check(a.model,a.checkpoint,a.code)
    except Exception as e:result={'pass':False,'error':str(e)}
    print(json.dumps(result));raise SystemExit(0 if result['pass'] else 1)

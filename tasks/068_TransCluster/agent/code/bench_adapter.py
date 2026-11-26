"""Model serialization and the unchanged author's one clustering step."""
import json
from pathlib import Path
import torch
from kmt import KModel

def model_seeds(seed):
    base=[153476998,235510766,205084180]
    return [int((s+1000003*(seed-1)) % (2**31-1)) for s in base]

def load_model(model_dir,index):
    files=json.loads((Path(model_dir)/'models.json').read_text())['files']
    ck=torch.load(Path(model_dir)/files[index],map_location='cpu',weights_only=False)
    a=ck['cli_args'];d=a['ndims']+a['nclusters']
    model=KModel(d,a['dqkv']*d,inv_temp=a['attn_itemp'],dropout_p=a['dropout'],act=a['attn_act'])
    model.load_state_dict(ck['model_state_dict']);model.eval()
    return model

def step(model,samples,centers):
    b,n,d=samples.shape;k=centers.shape[1]
    xx=torch.cat((samples.clone().detach(),torch.zeros(b,n,k)),dim=2)
    cc=torch.cat((centers.clone().detach(),torch.eye(k).repeat(b,1,1)),dim=2)
    _,cc=model(xx,cc)
    return cc[:,:,:d]

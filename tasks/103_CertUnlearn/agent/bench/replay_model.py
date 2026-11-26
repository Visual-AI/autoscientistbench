"""Independent trusted forward pass, preserving the released evaluation transforms."""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def channels_last(x):return x.permute(1,2,0)
def collate(batch):return np.asarray(torch.stack(batch))
class Images(torch.utils.data.Dataset):
    def __init__(self,x,augment):
        self.x=x
        aug=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()] if augment else []
        self.transform=transforms.Compose(aug+[transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),transforms.Lambda(channels_last)])
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return self.transform(Image.fromarray(self.x[i]))

def replay(model_dir,input_path,pred,seed,code=None):
    import multiprocessing as mp
    import os
    from pathlib import Path
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parent)
    mp.set_start_method('forkserver',force=True)
    mp.set_forkserver_preload(['replay_model'])
    import jax,jax.numpy as jnp
    from flax import serialization
    from model_def import CIFAR10TinykNet
    from certificate import verify_model_params
    torch.set_num_threads(1);torch.manual_seed(seed)
    with np.load(input_path,allow_pickle=False) as f:x=f['x'];augment=bool(f['augment'])
    loader=torch.utils.data.DataLoader(Images(x,augment),batch_size=128,shuffle=False,num_workers=4,collate_fn=collate)
    model=CIFAR10TinykNet(num_classes=10)
    @jax.jit
    def infer(p,x):return jnp.argmax(model.apply({'params':p},x,train=False),axis=-1)
    for epoch in range(1,51):
        state=serialization.msgpack_restore((model_dir/f'epoch_{epoch:02d}.msgpack').read_bytes())
        verify_model_params(state['params'])
        params=jax.tree_util.tree_map(jnp.asarray,state['params'])
        j=0
        for batch in loader:
            got=np.asarray(infer(params,batch))
            if not np.array_equal(got,pred[epoch-1,j:j+len(got)]):raise ValueError(f'prediction mismatch at epoch {epoch}')
            j+=len(got)

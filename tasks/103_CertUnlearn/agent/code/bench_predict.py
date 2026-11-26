"""Deferred epoch inference; images are provided only via --input."""
import argparse,csv,json,os
from pathlib import Path

def main():
    from runtime import configure_multiprocessing
    configure_multiprocessing()
    import numpy as np,torch,jax,jax.numpy as jnp
    from flax import serialization
    from src.models.model import ModelFactory
    from src.data.cifar_dataset import CIFAR10DataModule
    from src.data.base_datamodule import ArrayDataset,NumpyLoader
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.model/'certificate.json').read_text());config=c['config']
    torch.set_num_threads(1);torch.manual_seed(int(os.environ['BENCH_SEED']))
    with np.load(a.input,allow_pickle=False) as f:ids=f['id'];x=f['x'];augment=bool(f['augment'])
    dm=CIFAR10DataModule(config,None)
    dataset=ArrayDataset(x,np.zeros(len(x),dtype=np.int64),dm.train_transform if augment else dm.test_transform)
    loader=NumpyLoader(dataset,batch_size=128,shuffle=False,num_workers=4)
    model=ModelFactory.create_model(config['model']['name'],config['model']['n_classes'])
    @jax.jit
    def predict(params,x):return jnp.argmax(model.apply({'params':params},x,train=False),axis=-1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['id','epoch','pred'])
        for epoch in range(1,51):
            state=serialization.msgpack_restore((a.model/f'epoch_{epoch:02d}.msgpack').read_bytes())
            params=jax.tree_util.tree_map(jnp.asarray,state['params']);j=0
            for images,_ in loader:
                pp=np.asarray(predict(params,images))
                w.writerows((str(i),epoch,int(v)) for i,v in zip(ids[j:j+len(pp)],pp));j+=len(pp)
            assert j==len(ids)
            print(f'predicted epoch {epoch}',flush=True)
        f.write('#certificate='+json.dumps(c,sort_keys=True)+'\n')
if __name__=='__main__':main()

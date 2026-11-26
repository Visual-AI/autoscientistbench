"""The reproduced PABI experiment, with benchmark paths, seeds and saved epoch states."""
import argparse,hashlib,json,os,random
from pathlib import Path

def main():
    from runtime import configure_multiprocessing
    configure_multiprocessing()
    import numpy as np
    import torch
    import jax,flax,yaml
    from flax import serialization
    from src.data.datamodule import get_datamodule
    from src.models.model import ModelFactory
    from src.training.trainer import Trainer
    flax.config.update('flax_use_orbax_checkpointing',False)
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    seed=int(os.environ['BENCH_SEED']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1)
    config=yaml.safe_load((Path(__file__).parent/'config.yaml').read_text())
    config['global_seed']=seed
    data=Path(os.environ['BENCH_DATA_DIR']).resolve()
    config['dataset']['root']=str(data);config['training']['pretrained']=str(data/'train/base')
    config['checkpoint_path']=str(out/'ckpt');config['wandb']['log_dir']=str(out)
    dm=get_datamodule(config,None);dm.setup()
    model=ModelFactory.create_model(config['model']['name'],config['model']['n_classes'])
    trainer=Trainer(config=config,model=model,key=jax.random.PRNGKey(seed),train_dataloader=dm.train_dataloader(),val_dataloader=dm.val_dataloader(),test_dataloader=None,forget_dataloader=dm.forget_dataloader(),retain_dataloader=dm.retain_dataloader(),use_pretrained=config['training']['pretrained'],run_dir=str(out))
    state=trainer.fit();state=trainer.unlearn(state)
    (out/'final_state.msgpack').write_bytes(serialization.to_bytes(state))
    # Sanitize paths from the public evidence record.
    config['dataset']['root']='data';config['training']['pretrained']='train/base';config.pop('checkpoint_path');config.pop('wandb')
    c=dict(format='pabi-certificate-v1',config=config,base_sha256=hashlib.sha256((data/'train/base/checkpoint_100').read_bytes()).hexdigest(),steps_per_epoch=trainer.unlearn_steps_per_epoch,final_step=int(state.step),noise=[json.loads(l) for l in (out/'noise_audit.jsonl').read_text().splitlines()],epoch_sha256=[hashlib.sha256((out/f'epoch_{e:02d}.msgpack').read_bytes()).hexdigest() for e in range(1,51)])
    (out/'certificate.json').write_text(json.dumps(c,sort_keys=True)+'\n')
if __name__=='__main__':main()

"""Protocol output adapter: fresh per-instance fit and full varying-tempering sampling."""
import json, time, random
from pathlib import Path
import numpy as np
import torch
from cpu_transformation_opt import inversetransform
class Stage:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
    def start(self,name):
        self.name=name;self.begin=time.time()
        print('START',name,flush=True)
    def end(self):
        print('END',self.name,time.time()-self.begin,flush=True)
def finish_run(cfg,output_folder,stage,model,ema,diffusion,field,targetsample,normalize,bounds,means,covariance_matrix,loss_hist):
    out=Path(output_folder);D=cfg.exp.d
    shared_rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),r_model=field.r_model.state_dict(),
                    X=field.X,log_p_w_X=field.log_p_w_X,tau_quantile=field.tau_quantile,
                    tau_mean=field.tau_mean,loss_hist=loss_hist),out/'trained.pt')
    torch.set_rng_state(shared_rng[0]);np.random.set_state(shared_rng[1]);random.setstate(shared_rng[2])
    stage.start('varying_langevin')
    with torch.no_grad():
        sample=diffusion.sample_annealed_langevin(ema,cfg.plot.nsamples,joint=0,
            score_scaled=True,tempfield=field,T=cfg.method.langevin_T,epsilon=cfg.method.langevin_epsilon)
    stage.end()
    stage.start('varying_inverse_transform')
    if cfg.data.transform2unitcube:
        sample=inversetransform(sample,cfg,bounds,means,covariance_matrix,torch.tensor(cfg.exp.mixture_probs))
    assert torch.isfinite(torch.as_tensor(sample)).all()
    np.save(out/'samples.npy',np.asarray(sample));stage.end()
    return sample

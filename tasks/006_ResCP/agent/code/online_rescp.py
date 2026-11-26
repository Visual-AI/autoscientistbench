"""Incremental equivalent of the released reservoir and CPU sampler.

Only previously released residuals enter the reservoir or sampling pool. The
original window+1 convention, draw counts, quantiles and eta=0 are retained.
"""
import numpy as np
import torch
from bench_main import Reservoir, tensor, EPS
from reservoir_conformal_prediction.src.utils.utils import minimize_intervals
from sampler_slices import sorted_quantiles
import sampling_core

class OnlineResCP:
    def __init__(self,obj,seed):
        self.obj=obj;self.cfg=cfg=obj['cfg']
        self.r=None;self.states=None;self.raw_state=None
        self.reservoir=object.__new__(Reservoir)
        self.reservoir.__dict__.update(_n_internal_units=cfg['reservoir']['n_internal_units'],
            _noise_level=cfg['reservoir']['noise_level'],_leak=cfg['reservoir']['leak'],
            _internal_weights=obj['internal'],_input_weights=obj['input_weights'])
        # All released settings have eta=0: current truth cannot affect an interval.
        if cfg['sampler']['eta'] != 0: raise ValueError('released online adapter requires eta=0')
        nodes=obj['cal_residual'].shape[1]
        cls=sampling_core.ConformalResidualSampler if nodes==1 else sampling_core.ConformalResidualSamplerParallel
        self.sampler=cls(cal_residuals=obj['cal_residual'].clone(),reservoir=None,alpha=.1,**cfg['sampler'])
        self.scalar=nodes==1;self.nodes=nodes
        window=self.sampler.past_residuals_window
        self.pool_states=obj['cal_states'].clone()[-window:]
        self.pool_residual=obj['cal_residual'].clone()[-window:]
        self.initial=len(self.pool_residual);self.draws=window;self.step=0;self.previous=None
        torch.manual_seed(seed)

    def history(self,start,residual):
        clipped=np.clip(residual,self.obj['lower'],self.obj['upper']).astype(np.float32)
        if self.r is None:
            if start!=0 or len(clipped)<1: raise ValueError('invalid initial history')
            self.r=tensor(clipped)
            self.states=torch.zeros((1,self.nodes,self.cfg['reservoir']['n_internal_units']))
            z=(self.r[1:]-tensor(self.obj['bias']))/(tensor(self.obj['scale'])+EPS)
        else:
            if start!=len(self.r): raise ValueError('noncontiguous released history')
            self.r=torch.cat([self.r,tensor(clipped)])
            z=(tensor(clipped)-tensor(self.obj['bias']))/(tensor(self.obj['scale'])+EPS)
        if len(z):
            z=torch.where(z.isnan(),torch.tensor(0.),z)
            # The original implementation draws all zero-noise reservoir values
            # before reseeding the sampler. Do not consume its random stream.
            with torch.random.fork_rng():
                raw=self.reservoir.get_states(z.permute(1,0,2),bidir=False,initial_state=self.raw_state).permute(1,0,2)
            self.raw_state=raw[-1].clone()
            norm=torch.linalg.norm(raw,dim=-1,keepdim=True)
            states=torch.where(~torch.isclose(norm,torch.zeros_like(norm)),raw/norm,raw)
            self.states=torch.cat([self.states,states])
        if self.previous is not None:
            t,state,point=self.previous
            if t>=len(self.r):raise ValueError('previous residual not yet observable')
            self.pool_states=torch.cat([self.pool_states,state[None]])
            self.pool_residual=torch.cat([self.pool_residual,(point+self.r[t:t+1])-point])
            self.previous=None

    @torch.no_grad()
    def predict(self,t,point):
        x=self.states[t-self.cfg['delay']-1]
        lo=max(0,self.initial+self.step-self.sampler.past_residuals_window-1)
        hi=self.initial+self.step
        states=self.pool_states[lo:hi];pool=self.pool_residual[lo:hi].squeeze()
        test=x.reshape(1,-1) if self.scalar else x[None]
        probabilities=self.sampler.compute_similarity(states,test,T=self.sampler.T)
        samples=self.sampler.sample_residuals(pool,probabilities,self.draws)
        count=self.sampler.n_quantiles
        if count>1:
            lb=torch.linspace(.001,self.sampler.alpha-.001,count);ub=1-self.sampler.alpha+lb
        else:
            lb=torch.tensor([self.sampler.alpha/2]);ub=torch.tensor([1-self.sampler.alpha/2])
        q=sorted_quantiles(samples,torch.cat([lb,ub]))
        quantiles=q[None] if self.scalar else q.T[None]
        point=tensor(point)[None]
        low,high=minimize_intervals(quantiles,point,point)
        if hi-lo<=self.sampler.past_residuals_window:self.draws+=1
        self.previous=(t,x.clone(),point.clone());self.step+=1
        return low.numpy().reshape(-1),high.numpy().reshape(-1)

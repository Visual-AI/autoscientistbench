"""Observation-only protocol adapter; all optimization is inherited from the official code."""
import argparse,csv,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

CONFIG = dict(inputdim=50,batch_size=400,lr=1e-3,l1_lambda=1e-9,
              dataset="nonparanormal",encodertype="mlp",iterations=5000)

def fit(job):
    ident,x,trial,cfg,run_seed = job
    import torch
    torch.set_num_threads(1)
    from observation_loader import ObservationLoader
    from ToyLatticeROC import ToyROCFlowExperiment,get_allgates
    class ObservedExperiment(ToyROCFlowExperiment):
        def _init_loader(self):
            self.inputdim=self.config["inputdim"]
            self.iter_starttime=None
            # Same RNG consumption as the two original MultivariateNormal samples.
            torch.empty((2048,50)).normal_()
            torch.empty((1000,50)).normal_()
            return ObservationLoader(x,self.config["batch_size"],self.device)
    seed = (10*(run_seed-1)+trial) % (2**32)
    exp=ObservedExperiment(cfg,seed=seed)
    exp.train(num_epochs=cfg["iterations"],viz_interval=1000)
    gates=get_allgates(exp.models["encoder"],50,exp.device)
    return ident,gates

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train");t.add_argument("--out",required=True)
    q=sub.add_parser("predict")
    for n in ("model","input","output"):q.add_argument("--"+n,required=True)
    a=p.parse_args();seed=int(os.environ.get("BENCH_SEED","1"));cpus=int(os.environ.get("BENCH_CPUS","8"))
    if a.cmd=="train":
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/"config.json").write_text(json.dumps(dict(config=CONFIG,bench_seed=seed)))
        return
    cfg=json.loads((Path(a.model)/"config.json").read_text())["config"]
    with np.load(a.input,allow_pickle=False) as z:
        # Trial is a public algorithm replicate index, never a data-generation seed.
        jobs=[(str(i),x,int(t),cfg,seed) for i,x,t in zip(z["id"],z["x"],z["trial"])]
    with ProcessPoolExecutor(max_workers=min(cpus,len(jobs)),mp_context=multiprocessing.get_context("spawn")) as pool:
        results=list(pool.map(fit,jobs))
    with open(a.output,"w",newline="") as f:
        w=csv.writer(f);w.writerow(["id"]+["g%d_%d"%(i,j) for i in range(50) for j in range(50)])
        for ident,gates in results:w.writerow([ident]+gates.astype(float).ravel().tolist())

if __name__=="__main__":main()

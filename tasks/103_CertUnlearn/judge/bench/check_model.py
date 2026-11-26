#!/usr/bin/env python3
"""Recompute privacy, replay all noise draws, bind epoch files to the prediction certificate.
Run in the isolated candidate execution environment; training claims are not a proof of execution.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from certificate import verify,verify_model_params

def main():
    a=argparse.ArgumentParser();a.add_argument('model',type=Path);a.add_argument('--input');a.add_argument('--pred');a.add_argument('--code');args=a.parse_args()
    try:
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:int(os.environ.get("BENCH_CPUS","4"))])
        import numpy as np
        import jax,jax.numpy as jnp
        from flax import serialization
        from functools import partial
        c=json.loads((args.model/'certificate.json').read_text());r=verify(c)
        @partial(jax.jit,static_argnums=(2,))
        def replay(key,sigma,shapes):
            next_key,*keys=jax.random.split(key,len(shapes)+1)
            return next_key,[sigma*jax.random.normal(k,shape,jnp.float32) for k,shape in zip(keys,shapes)]
        for n in c['noise']:
            key,draws=replay(jnp.asarray(n['input_key'],dtype=jnp.uint32),jnp.float32(n['sigma']),tuple(tuple(s) for s in n['noise_shapes']))
            assert np.asarray(key).tolist()==n['output_key'],'noise key transition'
            assert hashlib.sha256(b''.join(np.asarray(d).tobytes() for d in draws)).hexdigest()==n['noise_sha256'],'Gaussian replay mismatch'
        for e,digest in enumerate(c['epoch_sha256'],1):
            payload=(args.model/f'epoch_{e:02d}.msgpack').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==digest,'checkpoint digest mismatch'
            verify_model_params(serialization.msgpack_restore(payload)['params'])
        if args.pred:
            from prediction import read
            with np.load(args.input,allow_pickle=False) as f:ids=f['id']
            pred,pc=read(args.pred,ids)
            assert pc==c,'prediction/model certificate mismatch'
            from replay_model import replay
            replay(args.model,args.input,pred,int(c['config']['global_seed']),args.code)
        print(json.dumps(dict(ok=True,**r,noise_replayed=True,epoch_hashes_verified=50,predictions_replayed=bool(args.pred))))
        return 0
    except Exception as e:
        print(json.dumps(dict(ok=False,epochs_to_50pct=float('inf'),reason=str(e))));return 1
if __name__=='__main__':sys.exit(main())

"""Raw-image protocol adapter. IsoCLIP calculations are in iso_core.py."""
import time
START = time.monotonic()
import argparse, os, sys, json, zipfile, tempfile, shutil, subprocess, struct
from pathlib import Path

# One encoder job: instance index, 0=gallery/1=query, batch start.
JOB=struct.Struct('<3q')

def map_images(p):
    # Inputs are ZIP_STORED: map the bytes so that all encoders share one page cache copy.
    import numpy as np
    with zipfile.ZipFile(p) as z:info=z.getinfo('images.npy')
    if info.compress_type!=zipfile.ZIP_STORED:return np.load(p,allow_pickle=False)['images']
    with open(p,'rb') as f:
        f.seek(info.header_offset+26);name,extra=struct.unpack('<HH',f.read(4));f.seek(name+extra,1)
        np.lib.format.read_magic(f);shape,_,dtype=np.lib.format.read_array_header_1_0(f)
        return np.memmap(p,dtype=dtype,mode='r',offset=f.tell(),shape=shape)

def worker(files, dest, threads, jobs=None):
    import random, io
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from encoder_api import load_encoder, encode as encode_fixed
    from iso_core import apply_iso
    seed=int(os.environ.get('BENCH_SEED','1'))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    model,preprocess=load_encoder(None,Path(os.environ['BENCH_DATA_DIR'])/'weights/ViT-B-32.pt')
    model.eval().requires_grad_(False)
    if jobs is not None:
        # Encoder process: take fixed batches until the shared job file is exhausted.
        instances={}
        with torch.no_grad():
            while job:=os.read(jobs,JOB.size):
                i,which,start=JOB.unpack(job);p=Path(files[i])
                if i not in instances:d=np.load(p,allow_pickle=False);instances[i]=map_images(p),d['offsets'],d['rgb_first'],(d['gallery'],d['query'])
                blob,offsets,rgb,sets=instances[i];indices=sets[which]
                images=[]
                for idx in indices[start:start+32]:
                    im=Image.open(io.BytesIO(blob[offsets[idx]:offsets[idx+1]].tobytes()))
                    if rgb[idx]:im=im.convert('RGB')
                    images.append(preprocess(im))
                np.save(Path(dest)/'features'/f'{p.stem}.{which}.{start}.npy',encode_fixed(model,torch.stack(images)).cpu().numpy())
                if start%1024==0: print(p.stem,'encoded',start,'/',len(indices),flush=True)
        return
    _,projector=apply_iso(model.text_projection.data.T,model.visual.proj.T,150,50)
    with torch.no_grad():
        for filename in files:
            p=Path(filename);d=np.load(p,allow_pickle=False)
            q=d['query'];g=d['gallery']
            def encoded(which,indices):
                return torch.vstack([torch.from_numpy(np.load(Path(dest)/'features'/f'{p.stem}.{which}.{start}.npy')) for start in range(0,len(indices),32)])
            gf=encoded(0,g)
            # Same-set reuse follows the official feature cache behavior.
            if np.array_equal(q,g):qf=gf
            else:qf=encoded(1,q)
            gf=F.normalize(gf.float() @ projector);qf=F.normalize(qf.float() @ projector)
            similarities=torch.vstack([torch.matmul(x,gf.T).cpu() for x in qf.split(32)])
            np.save(Path(dest)/(p.stem+'.npy'),similarities.numpy())
            print(p.stem,'complete',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model');ap.add_argument('--input');ap.add_argument('--output');ap.add_argument('--worker',nargs='+');ap.add_argument('--dest');ap.add_argument('--threads',type=int);ap.add_argument('--jobs',type=int)
    a=ap.parse_args()
    if a.worker:worker(a.worker,a.dest,a.threads,a.jobs);return
    if not (Path(a.model)/'NO_TRAINING').is_file():raise ValueError('missing training-free marker')
    import numpy as np
    paths=sorted(Path(a.input).glob('*.npz'))
    if not paths:raise ValueError('no retrieval instances')
    cpus=int(os.environ.get('BENCH_CPUS','16'));n=min(len(paths),4,max(1,cpus//4));threads=max(1,min(4,cpus//n))
    # Largest processing jobs first; inputs have no dataset identities.
    sizes={p:len(np.load(p)['offsets']) for p in paths}
    groups=[[] for _ in range(n)];loads=[0]*n
    for p in sorted(paths,key=lambda p:-sizes[p]):
        i=loads.index(min(loads));groups[i].append(p);loads[i]+=sizes[p]
    with tempfile.TemporaryDirectory(prefix='iso-',dir=os.environ.get('TMPDIR')) as dest:
        # Same per-instance batches of 32 as encoding them in place; read through one shared file offset, so each goes to one process.
        (Path(dest)/'features').mkdir();jobs=tempfile.TemporaryFile(dir=dest)
        for i,p in enumerate(paths):
            d=np.load(p,allow_pickle=False);q=d['query'];g=d['gallery']
            for which,indices in enumerate([g] if np.array_equal(q,g) else [g,q]):
                for start in range(0,len(indices),32):jobs.write(JOB.pack(i,which,start))
        jobs.flush();os.lseek(jobs.fileno(),0,os.SEEK_SET)
        # Single-thread encoders first; the projector and matmuls then keep the original workers and thread count, whose bytes depend on it.
        for stage,stage_threads,extra in (([paths]*cpus,1,['--jobs',str(jobs.fileno())]),(groups,threads,[])):
            processes=[]
            for group in stage:
                env=os.environ.copy()
                for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]=str(stage_threads)
                processes.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',*map(str,group),'--dest',dest,'--threads',str(stage_threads),*extra],env=env,pass_fds=[jobs.fileno()]))
            codes=[p.wait() for p in processes]
            if any(codes):raise RuntimeError(f'workers failed {codes}')
        # Before the receipt: thousands of batch files must not delay process exit past it.
        shutil.rmtree(Path(dest)/'features')
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for p in paths:z.write(Path(dest)/(p.stem+'.npy'),arcname=p.stem+'.npy')
            # Receipt covers startup/imports/encoding/projection/output I/O; checked by external timing probe.
            import io
            b=io.BytesIO();np.save(b,np.array(json.dumps({'wall_s':time.monotonic()-START,'cpus':cpus,'seed':int(os.environ.get('BENCH_SEED','1'))})),allow_pickle=False)
            z.writestr('__meta__.npy',b.getvalue())

if __name__=='__main__':main()

"""Trusted observation materialization; never exposes seeds or X_star."""
import importlib.util,json,os,subprocess,sys,tarfile
from pathlib import Path
import generator
HERE=Path(__file__).resolve().parent
old=sys.modules.get('protocol');sys.modules['protocol']=generator._family
spec=importlib.util.spec_from_file_location('_trusted_tensor_family',HERE/'family/protocol30.py')
family=importlib.util.module_from_spec(spec);spec.loader.exec_module(family)
if old is not None:sys.modules['protocol']=old
else:sys.modules.pop('protocol',None)
SMALL_FIELDS=['n','n3','r','p','alpha','kappa_list','i_kappa','sigma_star','T','z0','z1','eta','decay','thresh_up','thresh_low','transform','Y','Omega','L','R','L_plus','R_plus','Xtrans','errors_ScaledGD','errors_GD']
TR_FIELDS=['As','states','n','n3','r','m','T','eta','thresh_up','thresh_low','kappa_list']
def extract_runtime(archive,directory):
    directory.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as f:
        for member in f.getmembers():
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe frozen runtime')
        f.extractall(directory)
    return directory

# MKL_CBWR: without it, two singular-vector signs of the truth generator depend on the node's CPU and the frozen labels stop matching.
def runtime_env(runtime,threads=2):
    env=dict(os.environ)
    env.update(OCTAVE_HOME=str(runtime/'octave'),LD_LIBRARY_PATH=str(runtime/'lib'),OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS=str(threads),MKL_CBWR='COMPATIBLE',REPRO_WORKER_THREADS=str(threads),MKL_DYNAMIC='FALSE',OMP_DYNAMIC='FALSE')
    return env

def launch(script,runtime,cpus=2,timeout=None,affinity=None):
    # Shares this process's CPUs; runtime_env(cpus) sets the thread count. affinity is kept for callers.
    cmd=[str(runtime/'octave/bin/octave-cli'),'--no-history','--no-init-file','--quiet',str(script)]
    with script.with_suffix('.log').open('w') as log:
        result=subprocess.run(cmd,env=runtime_env(runtime,cpus),stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError('trusted observation materialization failed')

def generate_unit(rows,kind,private_dir,public_dir,seed,records,smoke):
    private_dir.mkdir(parents=True,exist_ok=True);public_dir.mkdir(parents=True,exist_ok=True)
    obs=public_dir/'observations.bin';truth=private_dir/'truth.mat'
    if kind=='small':
        row=rows[0]
        text=generator._family.generate(row['base'],row['transform'],private_dir,index=row['index'],records=records,smoke=smoke)
        text=text.split('    %% Spectral initialization',1)[0]
        fields=SMALL_FIELDS
        end="\nsave('-mat7-binary','%s','X_star');\nend\n"%truth
    else:
        text=family.regression(private_dir,records=records,smoke=smoke).split('for t=1:T',1)[0]
        text+="\ntruths=struct(); for j=1:numel(states); st=states{j}; if st.method_index==1; name=sprintf('truth_%d_%d',st.transform_index,st.i_kappa); truths.(name)=st.X_star; end; states{j}=rmfield(st,'X_star'); end;\n"
        text+="save('-mat7-binary','%s','-struct','truths');\n"%truth
        fields=TR_FIELDS;end='\n'
    text=text.replace('20261022',str(int(seed)))
    names='{'+','.join("'%s'"%x for x in fields)+'}'
    text+="\nallowed=%s; keep={}; for j=1:numel(allowed); if exist(allowed{j},'var'); keep{end+1}=allowed{j}; end; end;\n"%names
    text+="save('-binary','%s',keep{:});\n"%obs+end
    script=private_dir/'materialize.m';script.write_text(text)
    meta={'kind':kind,'instances':[{k:r[k] for k in ('id','base','transform','index')} for r in rows],'records':records}
    (public_dir/'request.json').write_text(json.dumps(meta)+'\n')
    return script,truth

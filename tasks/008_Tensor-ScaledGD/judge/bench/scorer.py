"""Trusted preparation, tensor-error observations and original-stage timing."""
import concurrent.futures,json,os,sys,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from tensor_replay import PreparedEvaluator
from private_eval import remaining_time
from tensor_materialize import extract_runtime,generate_unit,launch
PID='058_Tensor-ScaledGD'
METRIC='scaledgd_iterations'
HERE=Path(__file__).resolve().parent

class Oracle:
    def __init__(self,truths,records):
        self.truths=truths;self.records=records;self.curves={};self.primary_closed=False
        self.previous={};self.certificates={};self.transcript=[]
    def __call__(self,method,args,kwargs):
        if method=='finish_primary':
            if args or kwargs or self.primary_closed or any(not self.curves.get((case,'ScaledGD')) for case in self.truths):raise ValueError('invalid primary closure')
            self.primary_closed=True;return True
        if method!='observe' or kwargs or len(args)!=4:raise ValueError('only observe is allowed')
        case,solver,iteration,value=args
        if type(case) is not int or case not in self.truths or solver not in ('ScaledGD','GD') or type(iteration) is not int:raise ValueError('invalid observation key')
        if solver=='ScaledGD' and self.primary_closed:raise ValueError('primary trajectory already sealed')
        key=(case,solver);curve=self.curves.setdefault(key,[])
        if iteration!=len(curve)+1 or iteration>self.records:raise ValueError('observation budget/order violation')
        value=np.asarray(value);truth=self.truths[case]
        if value.shape!=truth.shape or value.dtype.kind not in 'fci' or not np.isfinite(value).all():raise ValueError('invalid candidate tensor')
        error=float(np.linalg.norm((value-truth).ravel())/np.linalg.norm(truth.ravel()))
        curve.append(error);self.transcript.append((case,solver,iteration))
        if solver=='ScaledGD' and case not in self.certificates:
            if error<=1e-6:
                self.certificates[case]=(value.copy(),self.previous.pop(case,value).copy(),iteration-1)
            else:self.previous[case]=value.copy()
        return error


def serialize_original_payload(path,rows,traces,config,train_s,predict_started):
    """Keep original NPZ/certificate work inside the solve clock, privately."""
    payload={'ids':np.array([r['id'] for r in rows]),'config':np.array(config)}
    for i,row in enumerate(rows):
        oracle,case=traces[row['id']]
        payload[f'curve_{i}']=np.asarray(oracle.curves[(case,'ScaledGD')])
        if case in oracle.certificates:
            first,before,iteration=oracle.certificates[case]
        else:
            first=np.zeros(oracle.truths[case].shape);before=np.zeros_like(first);iteration=1000
        payload[f'iterations_{i}']=np.array(iteration)
        payload[f'first_{i}']=first;payload[f'before_{i}']=before
        gd=np.zeros(oracle.records);values=oracle.curves.get((case,'GD'),[]);gd[:len(values)]=values
        payload[f'gd_{i}']=gd
    with path.open('wb') as output:np.savez_compressed(output,**payload)
    # Legacy main.py captures its clock after compression, before appending wall_s.
    predict_s=time.monotonic()-predict_started;wall_s=train_s+predict_s
    with zipfile.ZipFile(path,'a') as archive:
        with archive.open('wall_s.npy','w') as output:np.lib.format.write_array(output,np.array(wall_s),allow_pickle=False)
    return wall_s,predict_s


def score_iterations(iterations):
    reached=all(x<1000 for x in iterations)
    return (sum(iterations) if reached else 32000),reached


def score(pred,private):
    private=Path(private);settings=json.loads((private/'replay.json').read_text())
    records=int(settings['records']);smoke=bool(settings.get('smoke_dimensions',False));overall_start=time.monotonic()
    cpus=sorted(os.sched_getaffinity(0))[:128];ntr=min(24,max(1,len(cpus)//2));smallcpus=cpus[ntr:] or cpus
    allocations=[smallcpus[i:i+2] for i in range(0,len(smallcpus),2)]
    # Only private operator smoke fixtures may reuse an existing large runtime.
    smoke_runtime=settings.get('smoke_runtime_dir') if smoke else None
    data_dir=Path(settings['smoke_data_dir']) if smoke and 'smoke_data_dir' in settings else HERE.parents[1]/'agent/data'
    with tempfile.TemporaryDirectory(prefix='tensor-private-') as folder:
        root=Path(folder)
        runtime=Path(smoke_runtime) if smoke_runtime else extract_runtime(private/'runtime.tar.gz',root/'runtime')
        system_libraries=[]
        for name in ('libblas.so.3','liblapack.so.3'):
            library=Path('/usr/lib/x86_64-linux-gnu')/name
            if library.is_symlink() and os.readlink(library).startswith('/etc/alternatives/'):
                system_libraries.append((str(library.resolve()),os.readlink(library)))
        proc_stub=root/'proc-compat';proc_stub.mkdir()
        (proc_stub/'exe').symlink_to('/ws/model/runtime/octave/bin/octave-cli')
        (proc_stub/'environ').write_bytes(b'LD_LIBRARY_PATH=/ws/model/runtime/lib\0')
        system_libraries.append((str(proc_stub),'/proc/self'))
        if smoke_runtime:system_libraries.append((str(runtime),'/opt/tensor-smoke-runtime'))
        with PreparedEvaluator(pred,PID,128,None,runtime_mounts=system_libraries,data_dir=data_dir) as ev:
            bootstrap_s=time.monotonic()-overall_start
            prepared=ev.prepare(cpus);train_s=prepared['elapsed_s']
            predict_started=time.monotonic()
            with np.load(private/'input.npz',allow_pickle=False) as z:rows=json.loads(str(z['instances']))
            with np.load(private/'labels.npz',allow_pickle=False) as z:
                ids=list(map(str,z['ids']));expected={name:z['truth_'+str(i)].copy() for i,name in enumerate(ids)}
            groups=[('small',[r]) for r in rows if r['base']!='TR']
            tr=[r for r in rows if r['base']=='TR']
            if tr:groups.append(('regression',tr))
            if not groups:raise ValueError('no evaluation cells')
            curves={};traces={};diagnostics=[]
            def unit(index,kind,part):
                work=root/str(index);allowed=work/'public';hidden=work/'private'
                seeds={int(r['seed']) for r in part}
                if len(seeds)!=1:raise ValueError('shared regression requires a shared frozen seed')
                script,truthfile=generate_unit(part,kind,hidden,allowed,next(iter(seeds)),records,smoke)
                assigned=cpus[:ntr] if kind=='regression' else allocations[index%len(allocations)]
                remaining=remaining_time()
                launch(script,runtime,len(assigned),remaining,affinity=assigned)
                truthdata=loadmat(truthfile);truths={};mapping={}
                for r in part:
                    case=0 if kind=='small' else (0 if r['transform']=='DFT' else 1)*4+int(r['index'])-1
                    field='X_star' if kind=='small' else 'truth_%d_%d'%((1 if r['transform']=='DFT' else 2),r['index'])
                    value=truthdata[field];truths[case]=value;mapping[case]=r['id']
                    if not smoke and not np.allclose(value,expected[r['id']],rtol=1e-10,atol=1e-12):raise ValueError('frozen generator truth mismatch')
                oracle=Oracle(truths,records)
                remaining=remaining_time()
                result=ev.run(allowed,handler=oracle,affinity=assigned,timeout_s=remaining)
                receipt=json.loads(Path(result['output_path']).read_text())
                if not oracle.primary_closed or receipt.get('completed') is not True:raise ValueError('incomplete solver')
                values={};records_by_name={}
                for case,name in mapping.items():
                    curve=oracle.curves.get((case,'ScaledGD'),[])
                    if not curve:raise ValueError('missing ScaledGD observations')
                    values[name]=curve;records_by_name[name]=(oracle,case)
                diag={'kind':kind,'cases':len(part),'rpc_calls':result['rpc_calls'],'elapsed_s':result['elapsed_s'],'allocated_cpus':len(assigned)}
                if smoke:diag['observation_order']=oracle.transcript
                return values,records_by_name,diag
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(25,len(groups),len(cpus))) as pool:
                futures=[pool.submit(unit,i,*g) for i,g in enumerate(groups)]
                for future in futures:
                    values,records_by_name,diag=future.result();curves.update(values);traces.update(records_by_name);diagnostics.append(diag)
            serialize_started=time.monotonic()
            wall,predict_s=serialize_original_payload(root/'legacy-result.npz',rows,traces,ev.configuration_text(),train_s,predict_started)
            serialization_s=time.monotonic()-serialize_started
    iterations=[]
    for r in rows:
        curve=curves[r['id']];found=next((i for i,x in enumerate(curve) if x<=1e-6),None)
        iterations.append(found if found is not None else 1000)
    value,reached=score_iterations(iterations)
    return {METRIC:value,'all_reached':reached,'wall_s':wall,'iterations':iterations,
            'protocol':'private-policy-v2','oracle_integrity':True,'private_state':True,'scope':'tiny-smoke' if smoke else 'full',
            'timing':{'train_replay_s':train_s,'predict_replay_s':predict_s,'serialization_s':serialization_s,
                      'infrastructure_bootstrap_s':bootstrap_s,'total_s':time.monotonic()-overall_start,
                      'scope':'trusted train replay plus observation generation, both solver trajectories and original NPZ serialization',
                      'submitted_train_seconds_used':False,'outer_model_used':False,
                      'boundary_limit':'full workload throughput is unverified'},
            'diagnostics':diagnostics,**({'smoke_curves':curves} if smoke else {})}
if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as exc:result={METRIC:float('inf'),'violation':str(exc)}
    print(json.dumps(result))

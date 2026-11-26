"""Original Octave solver arithmetic with observation-only initialization."""
import argparse,json,os,sys,tempfile
from pathlib import Path
from protocol30 import small,regression
from main import launch

def script_text(request,source,out,config):
    body="addpath('%s'); addpath('%s'); addpath('%s');\n"%(os.environ['BENCH_BENCH_DIR'],Path(__file__).parent/'tproduct',Path(__file__).parent/'cpu_opt')
    body+="load('%s');\n"%(source/'observations.bin')
    for key in ['z0','z1','decay']:body+='%s=%.17g;\n'%(key,float(config[key]))
    if request['kind']=='small':
        row=request['instances'][0]
        text=small(row['base'],row['transform'],out,row['index'],records=request['records'])
        tail='    %% Spectral initialization'+text.split('    %% Spectral initialization',1)[1]
        scaled,gd=tail.split('    %% GD',1)
        old='error = tensor_fro(X - X_star)/tensor_fro(X_star);'
        assert old in scaled and old in gd
        scaled=scaled.replace(old,"error = bench_observe(X,t,'ScaledGD',0);")
        gd=gd.replace(old,"error = bench_observe(X,t,'GD',0);")
        eta=config['eta_rtc'] if row['base']=='RTC' else config['eta']
        body+='eta=%.17g;\nfor private_case=1:1\n'%float(eta)
        body+=scaled+'    %% GD'+gd+"\nbench_observe([],0,'finish_primary',-1);\n"
    else:
        text=regression(out,records=request['records'])
        tail='for t=1:T'+text.split('for t=1:T',1)[1]
        old='error=tensor_fro(X-st.X_star)/tensor_fro(st.X_star);'
        assert old in tail
        tail=tail.replace(old,"methods={'ScaledGD','GD'}; error=bench_observe(X,t,methods{st.method_index},(st.transform_index-1)*4+st.i_kappa-1);")
        body+='eta=%.17g; initialization_s=0; optimization_clock=tic;\n'%float(config['eta'])
        # Preserve the original shared-RHS batching: ScaledGD and GD states are
        # advanced together on every t, with their own original stopping rules.
        body+=tail+"\nbench_observe([],0,'finish_primary',-1);\n"
    assert 'X_star' not in body,'truth must not appear in solver workspace'
    return body

def main():
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    source=Path(a.input);request=json.loads((source/'request.json').read_text());model=Path(a.model)
    config=json.loads((model/'config.json').read_text());cpus=sorted(os.sched_getaffinity(0))[:int(os.environ['BENCH_CPUS'])]
    with tempfile.TemporaryDirectory(prefix='solver-') as folder:
        out=Path(folder);script=out/'run.m';script.write_text(script_text(request,source,out,config))
        if request['kind']=='regression':cpus=cpus[:min(24,max(1,len(cpus)))]
        try:launch(script,model/'runtime',cpus,request['kind']=='regression',out/'run.log')
        except Exception:
            print((out/'run.log').read_text()[-5000:],file=sys.stderr,flush=True)
            raise
    Path(a.output).write_text(json.dumps({'completed':True,'kind':request['kind']})+'\n')
if __name__=='__main__':main()

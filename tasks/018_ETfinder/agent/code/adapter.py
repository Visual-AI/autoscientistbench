import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])
from eval_curated_overlap import reconstruct_candidates
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert (Path(a.model)/'model.ready').exists()
request=json.loads(Path(a.input).read_text());train=Path(os.environ['BENCH_DATA_DIR'])/'train';work=Path(a.output).parent/'method';work.mkdir(exist_ok=True)
records=train/'records.json'
if request['mode']=='smoke':
 records=work/'records.json';records.write_text(json.dumps(request['records']))
with (work/'pipeline.log').open('w') as log:
 subprocess.run([sys.executable,'main.py','-w',str(work/'ranking'),'-j',str(records),'-s','LDDEIPF','--mode','rep','-c','0.90','-a','0.90','-m','JTT','--ssap-k','8','--iqtree-bin','iqtree3','-t','Rhodobacter sphaeroides'],check=True,stdout=log,stderr=subprocess.STDOUT)
with (work/'funnel.log').open('w') as log:
 subprocess.run([sys.executable,'funnel.py',str(records),str(train/'assemblies.dat'),str(work/'funnel'),str(train/'16S')],check=True,stdout=log,stderr=subprocess.STDOUT)
candidates,_,_=reconstruct_candidates(work/'funnel');assemblies=[r['assembly'] for r in candidates]
assert len(set(assemblies))==len(assemblies)
Path(a.output).write_text(json.dumps({'assemblies':assemblies})+'\n')

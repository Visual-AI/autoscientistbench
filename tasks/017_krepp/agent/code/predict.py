"""Protocol wrapper: pass existing reads through the candidate krepp executable."""
import argparse,gzip,json,os,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=Path(a.input); data=Path(os.environ['BENCH_DATA_DIR'])/'train'
manifest=json.loads((source/'manifest.json').read_text())
if os.environ.get('BENCH_SMOKE')=='1':
    if not manifest.get('smoke_fixture'):raise ValueError('explicit small smoke fixture required')
    data=source
elif manifest.get('smoke_fixture'):
    raise ValueError('smoke fixtures cannot replace the full default task')
ids=[]
with gzip.open(source/'reads.fq.gz','rt') as f:
    while True:
        h=f.readline()
        if not h:break
        ids.append(h[1:].strip());f.readline();f.readline();f.readline()
assert len(ids)==manifest['n_reads'] and len(set(ids))==len(ids)
with tempfile.TemporaryDirectory(prefix='krepp-') as td:
    tab=Path(td)/'placements.tsv'
    cmd=[str(Path(a.model)/'krepp'),'place','-i',str(data/'index'),'-q',str(source/'reads.fq.gz'),'--num-threads',os.environ['BENCH_CPUS'],'--leave-out-refs',str(data/'excluded.txt'),'--no-multi','--tabular','-o',str(tab)]
    subprocess.run(cmd,check=True)
    placed={}
    with tab.open() as f:
        for line in f:
            if line.startswith('#'):continue
            v=line.rstrip('\n').split('\t')
            if not v or v[0]=='SEQ_ID':continue
            if v[0] in placed:raise ValueError('multiple placement rows for one read')
            placed[v[0]]=int(v[2])
    if set(placed)-set(ids):raise ValueError('unexpected read in placement output')
    with open(a.output,'w') as f:
        f.write('read_id\tedge\n')
        for rid in ids:f.write(rid+'\t'+str(placed.get(rid,-1))+'\n')

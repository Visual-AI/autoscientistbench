import argparse,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 inp=Path(a.input);model=Path(a.model);output=Path(a.output);work=output.parent/'cwl_work';work.mkdir(exist_ok=True)
 pools=json.loads((inp/'manifest.json').read_text())['pools'];predictions={};root=Path(__file__).resolve().parent
 env=os.environ.copy();env['PATH']=str(root/'bin')+':'+env['PATH'];env['PATCHER_MINIMAP2_PATH']=str(root/'bin/minimap2_old')
 for row in pools:
  pool=row['pool'];dest=work/pool;dest.mkdir(exist_ok=True);site=row['site']
  job={'fastqfile':{'class':'File','path':str(inp/row['file'])},'reffile':{'class':'File','path':str(inp/f'reference_{site}_bench.fa')},'config':{'class':'File','path':str(inp/f'config_{site}_bench.txt')},'data_type':row['data_type'],'sample_name':pool}
  job_path=dest/'input.json';job_path.write_text(json.dumps(job))
  with (dest/'cwl.log').open('w') as log:
   subprocess.run(['cwltool','--no-container','--quiet','--preserve-entire-environment','--outdir',str(dest/'out'),str(model/'cwl/aav_per_sample_workflow.cwl'),str(job_path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  hits=list((dest/'out').rglob(f'readname_{pool}.txt'))
  if len(hits)!=1:raise ValueError(f'{pool}: expected one classified-read output, got {hits}')
  found={}
  for line in hits[0].read_text().splitlines():
   if not line or line.startswith('#'):continue
   uid,label=line.split('\t')[:2]
   if uid in found:raise ValueError('duplicate classified read')
   found[uid]=label
  if not set(found)<=set(row['ids']):raise ValueError('unknown classified read')
  # Count all supplied reads; filtered reads cannot disappear from the denominator.
  predictions[pool]={uid:found.get(uid,'Unclassified') for uid in row['ids']}
  print(f'{pool}: {len(found)} classified, {len(row["ids"])-len(found)} filtered/unclassified',flush=True)
 output.write_text(json.dumps({'pools':predictions},separators=(',',':'))+'\n')
if __name__=='__main__':main()

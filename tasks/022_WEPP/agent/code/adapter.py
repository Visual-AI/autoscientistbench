"""Run existing surveillance analysis on already supplied observations."""
import argparse,json,os,subprocess,shutil,csv
from pathlib import Path
import concurrent.futures
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);args=p.parse_args()
code=Path(__file__).resolve().parent;inp=Path(args.input);output=Path(args.output);reference=Path(os.environ['BENCH_DATA_DIR'])/'train/reference';manifest=json.loads((inp/'manifest.json').read_text());cpus=int(os.environ['BENCH_CPUS']);jobs=min(len(manifest['samples']),max(1,cpus//4));per=max(1,cpus//jobs)
for filename in ['wepp','closest_peak_clustering']:
 assert (Path(args.model)/filename).is_file()
 shutil.copy2(Path(args.model)/filename,code/'build'/filename)
def one(sample):
 name=sample['id'];work=output.parent/'work'/name;data=work/'data/observed'/name;data.mkdir(parents=True,exist_ok=True)
 for key,mate in [('r1',1),('r2',2)]:
  source=(inp/sample[key]).resolve();assert source.is_relative_to(inp.resolve());(data/f'{name}_R{mate}.fastq.gz').symlink_to(source)
 (data/'tree.pb.gz').symlink_to(reference/'tree.pb.gz');shutil.copy2(reference/'reference.fa',data/'reference.fa')
 (work/'.snakemake').mkdir();(work/'.snakemake/.modified_freyja_installed').touch()
 env=os.environ.copy()
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','TBB_NUM_THREADS']:env[key]=str(per)
 cmd=['snakemake','-s',str(code/'workflow/Snakefile'),'--nolock','--config',f'DIR=observed/{name}',f'FILE_PREFIX={name}','TREE=tree.pb.gz','REF=reference.fa','CLADE_LIST=annotation_1','CLADE_IDX=0','MIN_PROP=0.005','DASHBOARD_ENABLED=False','--cores',str(per),'--rerun-incomplete']
 with (work/'analysis.log').open('w') as log:subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=work/'results/observed'/name/f'{name}_lineage_abundance.csv';rows=[]
 with result.open() as f:
  for lineage,value in csv.reader(f):rows.append([lineage,float(value)])
 assert rows,'analysis did not produce lineage abundance estimates'
 assert not list(work.rglob('*haplotypes.bam')),'optional sequence-bearing export unexpectedly ran'
 return {'id':name,'abundances':rows}
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:estimates=list(pool.map(one,manifest['samples']))
output.write_text(json.dumps({'samples':estimates},allow_nan=False)+'\n')

import argparse,csv,json,subprocess,sys
from pathlib import Path
import pysam
p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=Path(__file__).resolve().parent;inp=Path(a.input);manifest=json.loads((inp/'manifest.json').read_text());work=Path(a.output).parent/'tarpon_work';work.mkdir(exist_ok=True);result={}
for sample in manifest['samples']:
 name=sample['name'];folder=work/name;folder.mkdir(exist_ok=True)
 retained=folder/'retained.bam';no_start=folder/'no_start.bam';filtered=folder/'filtered.bam';stats=folder/'boundary.tsv'
 command=[sys.executable,str(source/'identify_telo_start.py'),'--input_file',str(inp/sample['file']),'--repeat','GGTTAG','--sliding_window','100','--sliding_window_interval','15','--upper_threshold','0.6','--lower_threshold','0.05','--telomeric_rep_perc','0.7','--consecutive_repeats','3','--consecutive_threshold','15','--telomeric_fastq_out',str(retained),'--no_telomere_out',str(no_start),'--filtered_out',str(filtered),'--stats_fh',str(stats),'--mutant','false','--pre_telomeric_repeat_percentage','0.1','--pre_telo_distance','2000','--minimum_telomere_length','0']
 with (folder/'method.log').open('w') as log:
  subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([sys.executable,str(source/'variantRepeatAnalysis.py'),'--input_file',str(retained),'--stats_file',str(stats),'--repeat','GGTTAG','--stats_out',str(folder/'composition.tsv'),'--repeat_distribution',str(folder/'distribution.tsv')],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows={}
 with stats.open() as f:
  for r in csv.DictReader(f,delimiter='\t'):rows[r['read_id']]={'status':'retained','start':int(r['vrr_start_pos'])}
 for path,status in [(no_start,'no_start'),(filtered,'filtered')]:
  with pysam.AlignmentFile(path,'rb',check_sq=False) as bam:
   for r in bam:
    if r.query_name in rows:raise ValueError('duplicate output read')
    rows[r.query_name]={'status':status,'start':None}
 if len(rows)!=sample['count']:raise ValueError('lost boundary-stage reads')
 result[name]=rows
Path(a.output).write_text(json.dumps({'samples':result},separators=(',',':'))+'\n')

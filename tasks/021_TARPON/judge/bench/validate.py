import sys,json,pysam
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read,check
pred=read(sys.argv[1]);inp=Path(sys.argv[2]);manifest=json.loads((inp/'manifest.json').read_text())
if set(pred)!={s['name'] for s in manifest['samples']}:raise ValueError('sample population changed')
for s in manifest['samples']:
 with pysam.AlignmentFile(inp/s['file'],'rb',check_sq=False) as bam:seq={r.query_name:r.query_sequence for r in bam}
 if len(seq)!=s['count']:raise ValueError('bad fixed input')
 check(pred[s['name']],seq)

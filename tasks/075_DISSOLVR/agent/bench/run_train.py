"""Time the complete Python training subprocess; the outer judge also times bash.
This helper is frozen. Calling it is a protocol rule checked by patch audit.
"""
import json,subprocess,sys,time
from pathlib import Path
start=time.monotonic()
code,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(code/'regime-i/train.py'),'--out',str(out)],check=True)
elapsed=time.monotonic()-start
usage=json.loads((out/'label_usage.json').read_text())
meta={'train_s':elapsed,'train_only':int(usage['label_sources']==['train/train.csv'])}
(out/'train_meta.json').write_text(json.dumps(meta)+'\n')

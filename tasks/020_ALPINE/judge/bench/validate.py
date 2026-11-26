import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
p=read(sys.argv[1]);m=json.loads((Path(sys.argv[2])/'manifest.json').read_text())
for row in m['pools']:
 if set(p[row['pool']])!=set(row['ids']):raise ValueError('read ids do not match input')

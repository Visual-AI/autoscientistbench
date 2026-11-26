import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
pred=read(sys.argv[1]);manifest=json.loads((Path(sys.argv[2])/'manifest.json').read_text())
if set(pred)!={s['id'] for s in manifest['samples']}:raise ValueError('missing or extra samples')

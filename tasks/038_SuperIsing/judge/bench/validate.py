import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
request=json.loads(Path(sys.argv[2]).read_text())
read(sys.argv[1],request['mode'])

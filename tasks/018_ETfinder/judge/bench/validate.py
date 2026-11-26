import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
read(sys.argv[1],json.load(open(sys.argv[2]))['allowed_assemblies'])

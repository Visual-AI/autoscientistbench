import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_predictions
pred=read_predictions(sys.argv[1])
with open(sys.argv[2],newline='') as f: rows=list(csv.DictReader(f))
ids=[r['id'] for r in rows]
if len(ids)!=len(set(ids)) or set(ids)!=set(pred): raise ValueError('prediction IDs do not match inputs')

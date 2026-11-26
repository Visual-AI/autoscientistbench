import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_predictions
elapsed,rows=read_predictions(sys.argv[1])
with open(Path(sys.argv[2])/'index.csv',newline='') as f:ids=[r['id'] for r in csv.DictReader(f)]
if len(ids)!=len(set(ids)) or set(ids)!=set(rows):raise ValueError('IDs mismatch')

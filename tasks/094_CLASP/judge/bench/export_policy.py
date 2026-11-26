"""Export a policy artifact; no evaluation data are accessed here."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import export_artifact
TASK_ID = 'icml-6078-CLASP'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 m=json.loads(Path(a.input).read_text())
 if m.get('task_id')!=TASK_ID or m.get('protocol')!='private-policy-v2': raise ValueError('wrong public input manifest')
 export_artifact(a.code,a.model,a.output,TASK_ID)

"""Check the sealed policy artifact and public manifest without executing code."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import validate_artifact
TASK_ID = 'icml-3011-AsymP-GDA'
if __name__=='__main__':
 try:
  m=json.loads(Path(sys.argv[2]).read_text())
  if m.get('task_id')!=TASK_ID or m.get('protocol')!='private-policy-v2': raise ValueError('wrong public input manifest')
  if not isinstance(m.get('ids'),list) or len(m['ids'])!=len(set(m['ids'])) or not m['ids']: raise ValueError('invalid ids')
  validate_artifact(sys.argv[1],TASK_ID)
  print('ok: sealed policy artifact')
 except Exception as exc:
  print('invalid: '+str(exc),file=sys.stderr);sys.exit(1)

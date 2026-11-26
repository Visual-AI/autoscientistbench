"""Validate the inert candidate archive; no candidate code runs here."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import validate_artifact
from protocol import TASK_ID

def validate(pred_path,input_path):
    request=json.loads(Path(input_path).read_text())
    if request.get('task_id')!=TASK_ID:raise ValueError('task mismatch')
    validate_artifact(pred_path,TASK_ID)

if __name__=='__main__':
    try:validate(*sys.argv[1:])
    except Exception:
        print('invalid policy artifact');sys.exit(1)
    print('valid private policy artifact')

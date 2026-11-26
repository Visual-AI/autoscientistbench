import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import validate_artifact
PID='058_Tensor-ScaledGD'
def check(pred,input_path):
    descriptor=json.loads(Path(input_path).read_text())
    if descriptor.get('task_id')!=PID:raise ValueError('wrong request')
    return validate_artifact(pred,PID)
if __name__=='__main__':
    try:check(*sys.argv[1:]);print('valid policy artifact')
    except Exception as e:sys.exit('invalid artifact: '+str(e))

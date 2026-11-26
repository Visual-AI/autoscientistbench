import hashlib,json,sys
from pathlib import Path
from private_eval import validate_artifact
def check(pred,inp):
 root=Path(inp);m=json.loads((root/'manifest.json').read_text())
 if m.get('protocol')!='private-active-label-v2' or m.get('task_id')!='imp-104-BalancedActiveInf' or 'oracle_sha256' in m or (root/'oracle.npy').exists():raise ValueError('invalid public episode')
 if hashlib.sha256((root/'predictions.npz').read_bytes()).hexdigest()!=m['predictions_sha256']:raise ValueError('frozen predictions changed')
 validate_artifact(pred,'imp-104-BalancedActiveInf')
if __name__=='__main__':
 try:check(*sys.argv[1:]);print('ok: private active-label policy')
 except Exception as e:sys.exit(str(e))

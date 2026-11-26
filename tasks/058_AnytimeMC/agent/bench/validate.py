import json,sys
from private_eval import validate_artifact
def check(pred,inp):
 d=json.load(open(inp))
 if set(d)!={'protocol','task_id','trials','M','B','alpha'} or d['protocol']!='private-null-values-v3' or d['task_id']!='Sta-33':raise ValueError('invalid protocol descriptor')
 if any(type(d[k]) is not int or d[k]<1 for k in ('trials','M','B')) or d['alpha']!=0.1:raise ValueError('invalid dimensions')
 validate_artifact(pred,'Sta-33')
if __name__=='__main__':
 try:check(*sys.argv[1:]);print('ok: private sampler policy')
 except Exception as e:sys.exit(str(e))

import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
m=json.loads((Path(a.model)/'simulation.json').read_text());request=json.loads(Path(a.input).read_text())
out={'mode':request['mode'],'fixed_couplings':m['fixed_couplings'],'cases':[{k:r[k] for k in ['nmf','tag','parameters']} for r in m['cases']]}
Path(a.output).write_text(json.dumps(out,allow_nan=False)+'\n')

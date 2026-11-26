import argparse,json,os
from pathlib import Path
import torch
from private_rpc import call
from online_rescp import OnlineResCP
p=argparse.ArgumentParser()
for name in ('model','input','output'):p.add_argument('--'+name,required=True)
a=p.parse_args();torch.set_num_threads(min(4,max(1,int(os.environ.get('BENCH_CPUS','1')))))
request=json.loads(Path(a.input).read_text());cell=request['cell']
obj=torch.load(Path(a.model)/(cell+'.pt'),map_location='cpu',weights_only=False)
policy=OnlineResCP(obj,int(os.environ.get('BENCH_SEED','1')))
while True:
    event=call('next')
    if event['op']=='done':break
    policy.history(event['history_start'],event['residual'])
    low,high=policy.predict(event['query'],event['point'])
    call('submit',{'lower':low,'upper':high})
Path(a.output).write_text('{"complete":true}\n')

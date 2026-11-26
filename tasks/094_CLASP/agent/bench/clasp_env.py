"""Causal RPC interface; a decision is committed before feedback is returned."""
import csv,json
import numpy as np
from private_rpc import call
N,K,T=10,4,100
F,G,D=N**2,np.sqrt(4*N),np.sqrt(2)
class Episode:
 def initial(self):return np.asarray(call('initial'))
 def step(self,x):return tuple(np.asarray(v) for v in call('step',np.asarray(x)))
 def step_many(self,xs):return [self.step(x) for x in xs]
 @property
 def rounds(self):return int(call('rounds'))
 def finish(self,final_x):return call('finish',np.asarray(final_x))
def make_episode(episode_id):call('start',str(episode_id));return Episode()
HEADER=['id','actions','rounds','loss','ccv1','ccv2']
def write_output(path,records):
 with open(path,'w',newline='') as f:
  w=csv.writer(f);w.writerow(HEADER)
  for key,rec in records:w.writerow([key,json.dumps(rec['actions'],separators=(',',':')),rec['rounds'],repr(rec['loss']),repr(rec['ccv1']),repr(rec['ccv2'])])

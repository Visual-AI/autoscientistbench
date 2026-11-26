"""Causal RPC API: only the trusted service owns future contexts and utilities."""
import numpy as np
from private_rpc import call
T,D,E,K=2000,10,20,10
class Episode:
 def get_context(self):return np.asarray(call('get_context'))
 def serve(self,a_idx,b_idx):return tuple(np.asarray(x) for x in call('serve',int(a_idx),int(b_idx)))
 def advance(self):return call('advance')
 def finish(self):return call('finish')
def make_episode(episode_id):
 call('start',str(episode_id));return Episode()
class EpisodeBatch:
 def __init__(self,episodes):self._episodes=episodes
 def get_context(self):return np.stack([e.get_context() for e in self._episodes])
 def serve(self,a,b):
  if len(a)!=len(self._episodes) or len(b)!=len(self._episodes):raise ValueError('one pair per episode required')
  out=[e.serve(x,y) for e,x,y in zip(self._episodes,a,b)]
  return np.stack([x[0] for x in out]),np.stack([x[1] for x in out])
 def advance(self):return [e.advance() for e in self._episodes]
 def finish(self):return [e.finish() for e in self._episodes]
def make_episodes(ids):return EpisodeBatch([make_episode(i) for i in ids])

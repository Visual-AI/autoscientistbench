"""Trusted state machine: future online losses never enter the candidate worker."""
from private_clasp import make_episode
class Handler:
 def __init__(self,row):
  self.id=row['id'];self.episode=make_episode(int(row['seed']));self.started=False;self.finished=False;self.invalid=False;self.record=None
 def __call__(self,method,args,kwargs):
  try:
   if kwargs:raise ValueError('keyword arguments are not accepted')
   if method=='start':
    if self.started or args!=[self.id]:raise ValueError('one start for the assigned episode required')
    self.started=True;return {'id':self.id}
   if not self.started or self.finished:raise ValueError('episode is not active')
   if method not in ('initial','step','rounds','finish'):raise ValueError('unknown oracle method')
   expected=1 if method in ('step','finish') else 0
   if len(args)!=expected:raise ValueError('wrong argument count')
   result=self.episode.rounds if method=='rounds' else getattr(self.episode,method)(*args)
   if method=='finish':self.finished=True;self.record=result
   return result
  except Exception:self.invalid=True;raise
 def accepted_record(self):
  if self.invalid or not self.finished:raise ValueError('incomplete or invalid trusted episode')
  return self.record

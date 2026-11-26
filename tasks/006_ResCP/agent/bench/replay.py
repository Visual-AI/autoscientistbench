"""Judge-owned cell history: reveal [previous cutoff, query), then commit interval."""
import numpy as np

class Replay:
    def __init__(self,arrays,cell):
        self.a={k.split('__',1)[1]:v for k,v in arrays.items() if k.startswith(cell+'__')}
        self.cell=cell;self.index=0;self.cutoff=0;self.waiting=False;self.failed=False;self.predictions=[]
        q=self.a['query']
        if len(q)==0 or np.any(np.diff(q)<=0) or q[0]<1 or q[-1]>len(self.a['residual']):raise ValueError('invalid query timeline')
    def __call__(self,method,args,kwargs):
        try:
            if self.failed or kwargs:raise ValueError('invalid temporal request')
            if method=='next' and not args:
                if self.waiting:raise ValueError('submit current interval before advancing')
                if self.index==len(self.a['query']):return {'op':'done'}
                t=int(self.a['query'][self.index]);start=self.cutoff;self.cutoff=t;self.waiting=True
                event={'op':'predict','query':t,'history_start':start,'residual':self.a['residual'][start:t],
                       'point':self.a['point'][self.index], 'id':self.a['id'][self.index].reshape(-1).tolist()}
                for key in ('target','date'):
                    if key in self.a:
                        data=self.a[key][start:t]
                        event[key]=data.tolist() if data.dtype.kind not in 'biufc' else data
                if 'encode_start' in self.a:event['encode_start']=int(self.a['encode_start'])
                return event
            if method!='submit' or len(args)!=1 or not self.waiting:raise ValueError('unexpected temporal operation')
            reply=args[0]
            if not isinstance(reply,dict) or set(reply)!={'lower','upper'}:raise ValueError('invalid interval response')
            ids=self.a['id'][self.index].reshape(-1).tolist()
            low=np.asarray(reply['lower'],dtype=float);high=np.asarray(reply['upper'],dtype=float)
            if low.shape!=(len(ids),) or high.shape!=low.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):raise ValueError('invalid interval bounds')
            self.predictions.extend(zip(ids,low.tolist(),high.tolist()));self.index+=1;self.waiting=False
            return {'accepted':True}
        except Exception:self.failed=True;raise
    def rows(self):
        if self.failed or self.waiting or self.index!=len(self.a['query']):raise ValueError('incomplete temporal transcript')
        return self.predictions

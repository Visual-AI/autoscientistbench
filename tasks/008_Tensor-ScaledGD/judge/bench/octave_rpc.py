"""Unprivileged persistent Octave bridge to the legal scalar-error oracle."""
import json,sys
import numpy as np
from private_rpc import call
for line in sys.stdin:
    try:
        q=json.loads(line)
        if q.get('method')=='finish_primary':
            result=call('finish_primary');print(float(result),flush=True);continue
        shape=tuple(map(int,q['shape']));n=int(np.prod(shape))
        if len(shape)!=3 or n>1000000 or n<=0:raise ValueError('shape')
        data=np.fromfile(q['file'],dtype='<f8')
        expected=n*(2 if q['complex'] else 1)
        if data.size!=expected:raise ValueError('tensor size')
        x=data[:n]
        if q['complex']:x=x+1j*data[n:]
        x=x.reshape(shape,order='F')
        result=call('observe',int(q['case']),str(q['method']),int(q['iteration']),x)
        print(format(float(result),'.17g'),flush=True)
    except Exception:
        print('ERROR',flush=True);sys.exit(1)

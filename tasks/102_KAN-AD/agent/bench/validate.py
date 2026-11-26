"""Validate the complete five-trial anomaly-score tensor."""
import sys
import numpy as np

def check(pred_path,input_path):
    with np.load(input_path,allow_pickle=False) as z:
        ids=z['ids'];lengths=np.diff(z['offsets'])-96
    with np.load(pred_path,allow_pickle=False) as p:
        if set(p.files)!={'ids','offsets','scores'}:raise ValueError('exact keys ids, offsets, scores required')
        got=p['ids'];off=p['offsets'];s=p['scores']
    if got.ndim!=1 or got.dtype.kind not in 'US':raise ValueError('ids must be a string vector')
    if len(set(got.tolist()))!=len(got):raise ValueError('duplicate ids')
    if set(got.tolist())!=set(ids.tolist()):raise ValueError('id set mismatch')
    if off.dtype.kind not in 'iu' or off.shape!=(len(got)+1,) or off[0]!=0:raise ValueError('invalid offsets')
    sizes=dict(zip(ids.tolist(),lengths.tolist()))
    if not np.array_equal(np.diff(off),[sizes[i] for i in got]):raise ValueError('incorrect score lengths')
    if s.shape!=(5,int(off[-1])) or s.dtype.kind!='f':raise ValueError('scores must be floating (5, total scored points)')
    if not np.isfinite(s).all():raise ValueError('nonfinite scores')
    return got,off,s

if __name__=='__main__':
    try:check(sys.argv[1],sys.argv[2])
    except Exception as e:sys.exit(str(e))
    print('ok')

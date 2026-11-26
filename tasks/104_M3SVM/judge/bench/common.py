"""Frozen prediction parser and privacy ledger checker."""
import csv,json,math
import numpy as np
RUNS=10
FIELDS=['id']+[f'history_{r}' for r in range(RUNS)]+[f'delta_{r}' for r in range(RUNS)]+[f'pred_{r}' for r in range(RUNS)]

def ledger(history):
    if isinstance(history,dict):
        if set(history)!={'accountant','segments'} or history['accountant']!='mechanisms-v2':
            raise ValueError('unknown mechanism accountant')
        segments=history['segments']
        if not isinstance(segments,list) or not 1<=len(segments)<=10000:
            raise ValueError('nonempty mechanism segments required')
        for segment in segments:
            if not isinstance(segment,dict):raise ValueError('mechanism segment must be an object')
            kind=segment.get('mechanism')
            fields={'gaussian':{'l2_sensitivity','noise_std','releases'},
                    'laplace':{'l1_sensitivity','noise_scale','releases'},
                    'poisson_gaussian':{'noise_multiplier','sample_rate','steps'}}.get(kind)
            if fields is None or set(segment)!=fields|{'mechanism'}:
                raise ValueError('unsupported mechanism or parameters')
            for key in fields:
                value=segment[key]
                if type(value) not in (int,float) or not math.isfinite(value):
                    raise ValueError('nonfinite mechanism parameter')
                if key in ('releases','steps'):
                    if value<1 or int(value)!=value:raise ValueError('positive integer composition count required')
                elif key.endswith('sensitivity'):
                    if value<0:raise ValueError('negative sensitivity')
                elif value<=0:raise ValueError('noise and sample rates must be positive')
            if kind=='poisson_gaussian' and segment['sample_rate']>1:
                raise ValueError('sample rate exceeds one')
        return history
    if not isinstance(history,list) or not history or len(history)>10000:raise ValueError('nonempty accountant history required')
    for row in history:
        if not isinstance(row,(list,tuple)) or len(row)!=3:raise ValueError('ledger rows must be sigma,q,steps')
        sigma,q,steps=row
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row):raise ValueError('nonfinite ledger')
        if sigma<=0 or not 0<q<=1 or steps<1 or int(steps)!=steps:raise ValueError('invalid ledger parameters')
    return history

def read_predictions(path):
    ids=[];pred=[];metadata=None;histories=None;deltas=None
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('incorrect prediction header')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('incorrect row width')
            idx=int(row['id'])
            if str(idx)!=row['id'] or idx<0:raise ValueError('id must be a canonical nonnegative integer')
            meta=[row[f'history_{r}'] for r in range(RUNS)]+[row[f'delta_{r}'] for r in range(RUNS)]
            if metadata is None:
                metadata=meta;histories=[ledger(json.loads(v)) for v in meta[:RUNS]];deltas=[float(v) for v in meta[RUNS:]]
                if not all(math.isfinite(v) and 0<v<1 for v in deltas):raise ValueError('invalid delta')
            elif meta!=metadata:raise ValueError('accountant ledger changes across examples')
            values=[float(row[f'pred_{r}']) for r in range(RUNS)]
            if not all(math.isfinite(v) and v==int(v) and 0<=v<26 for v in values):raise ValueError('predictions must be class integers 0..25')
            ids.append(idx);pred.append(values)
    if not ids or len(set(ids))!=len(ids):raise ValueError('empty prediction or duplicate id')
    return np.asarray(ids,dtype=np.int64),np.asarray(pred,dtype=np.int64),histories,deltas

def privacy(histories,deltas):
    if len(histories)!=RUNS or len(deltas)!=RUNS:raise ValueError('ten independent privacy ledgers required')
    eps=[]
    for h,d in zip(histories,deltas):
        if d!=1e-5:raise ValueError('delta must equal 1e-5')
        ledger(h)
        if isinstance(h,list):
            from opacus.accountants import RDPAccountant
            a=RDPAccountant();a.history=[tuple(row) for row in h]
            e=float(a.get_epsilon(delta=d))
        else:
            # Trusted bounds for declared mechanisms; actual sensitivities and noise
            # execution require the same training-provenance audit as legacy ledgers.
            orders=np.array([1.25,1.5,1.75,*range(2,65),128,256,512,1024],dtype=float)
            rdp=np.zeros_like(orders);pure=0.;all_pure=True
            for segment in h['segments']:
                kind=segment['mechanism']
                if kind=='laplace':
                    pure+=segment['releases']*segment['l1_sensitivity']/segment['noise_scale']
                elif kind=='gaussian':
                    all_pure=False
                    rdp+=segment['releases']*orders*(segment['l2_sensitivity']/segment['noise_std'])**2/2
                else:
                    from opacus.accountants.analysis.rdp import compute_rdp
                    all_pure=False
                    rdp+=compute_rdp(q=segment['sample_rate'],noise_multiplier=segment['noise_multiplier'],
                                     steps=int(segment['steps']),orders=orders)
            # Pure-DP composition is exact for this conservative bound. Otherwise
            # additive RDP composition and the standard RDP->(epsilon,delta) bound.
            e=float(pure if all_pure else pure+np.min(rdp+math.log(1/d)/(orders-1)))
        if not math.isfinite(e) or e<0 or e>1.0:raise ValueError('accountant epsilon exceeds 1.0')
        eps.append(e)
    return eps

#!/usr/bin/env python3
"""Frozen ARI and ranking guard; elapsed fit time is diagnostic."""
import json, math, statistics, sys
from common import load, predictions, ari, spearman
SPEARMAN_FLOOR = 0.8211555622531486

def score(pred_path, labels_path):
    try:
        labels=load(labels_path)
        truth=labels['datasets']
        if labels['scope']=='full':
            expected={(d,k,r) for d in (1,2,3) for k in (3,5,7) for r in range(1,11)}
            cells=[tuple(x['cell']) for x in truth]
            if len(cells)!=90 or set(cells)!=expected or any(len(x['partition'])!=105 for x in truth): raise ValueError('incomplete full protocol')
        elif labels['scope']=='smoke':
            if len(truth)!=1 or len(truth[0]['partition'])!=12: raise ValueError('invalid smoke fixture')
        else: raise ValueError('unknown label scope')
        pred=predictions(load(pred_path),[len(t['partition']) for t in truth])
        aris=[ari(p['partition'],t['partition']) for p,t in zip(pred,truth)]
        rhos=[spearman(p['strength'],t['strength']) for p,t in zip(pred,truth)]
        mean_ari=statistics.mean(aris); mean_rho=statistics.mean(rhos)
        median_time=statistics.median(p['fit_time_s'] for p in pred)
        rank_ok=mean_rho>=SPEARMAN_FLOOR
        return {'ari_minVI':mean_ari if rank_ok else 0.0,
                'raw_mean_ari':mean_ari,'mean_spearman':mean_rho,'median_fit_time_s':median_time,
                'ranking_guard':rank_ok,'n_datasets':len(pred)}
    except Exception as e:
        return {'ari_minVI':0.0,'invalid':True,'reason':str(e)}

if __name__=='__main__': print(json.dumps(score(*sys.argv[1:]),allow_nan=False))

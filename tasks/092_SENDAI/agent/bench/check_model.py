#!/usr/bin/env python3
"""Check sparse-observation certificate. Actual information use also requires patch review."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import validate

def check(model,inp,pred):
    card=json.loads((Path(model)/'model_card.json').read_text())
    assert card['format']=='sendai-sparse/v1'
    assert card['replicates']==3
    assert card['seeds']==[3*card['outer_seed']+i for i in range(3)]
    p=validate(pred,inp)
    loc=np.asarray(card['sensor_locs'])
    if loc.size==0:loc=loc.reshape(0,2)
    assert np.array_equal(loc,p['sensor_locs'])
    assert 0<=len(loc)<=64 and 0<=card['lags']<=5 and card['lags']==int(p['lags'])
    with np.load(inp,allow_pickle=False) as x:
        assert set(map(tuple,loc)).issubset(set(map(tuple,x['sensor_locs'])))
    return {'pass':True,'sensors':len(loc),'lags':card['lags']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model');ap.add_argument('--input',required=True);ap.add_argument('--pred',required=True);a=ap.parse_args()
    try:print(json.dumps(check(a.model,a.input,a.pred)))
    except Exception as e:print(json.dumps({'pass':False,'error':str(e)}));sys.exit(1)

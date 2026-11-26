"""Safe JSON representation of the fitted numeric encoder and five extracted rule sets."""
import csv
import json
# Certificates may exceed Python's default 128 KiB CSV-cell limit.
csv.field_size_limit(8 * 1024 * 1024)
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from frozen_encoder import TabularEncoder
from frozen_rules import RuleSet, predict_rules
FEATURES=['preg','plas','pres','skin','insu','mass','pedi','age']
COLUMNS=['id','certificate','pred_0','pred_1','pred_2','pred_3','pred_4']

def export_encoder(enc):
    return {'features':enc.continuous_features,'mean':enc._scaler.mean_.tolist(),
            'scale':enc._scaler.scale_.tolist(),'thresholds':[t.tolist() for t in enc._thresholds],
            'classes':enc.class_names,'num_bits':enc.num_bits}

def restore_encoder(d):
    assert d['features']==FEATURES,'encoder feature order mismatch'
    assert d['classes']==['tested_negative','tested_positive'],'class order mismatch'
    b=d['num_bits'];assert isinstance(b,int) and b>=1,'invalid bit count'
    mu=np.asarray(d['mean'],float);scale=np.asarray(d['scale'],float);th=np.asarray(d['thresholds'],float)
    assert mu.shape==scale.shape==(8,) and th.shape==(8,b),'encoder shape mismatch'
    assert np.isfinite(mu).all() and np.isfinite(scale).all() and np.isfinite(th).all() and (scale>0).all(),'invalid encoder numbers'
    enc=TabularEncoder(num_bits=b);enc.continuous_features=FEATURES.copy()
    enc._scaler=StandardScaler();enc._scaler.mean_=mu;enc._scaler.scale_=scale;enc._scaler.var_=scale**2;enc._scaler.n_features_in_=8
    enc._scaler.feature_names_in_=np.asarray(FEATURES,dtype=object)
    enc._thresholds=list(th);enc._label_enc=LabelEncoder();enc._label_enc.classes_=np.asarray(d['classes']);enc._fitted=True
    return enc

def unpack(card):
    assert card['format']=='tt-sparse-rules-v1','certificate format'
    assert len(card['models'])==5,'need five independent replicates'
    result=[]
    for d in card['models']:
        enc=restore_encoder(d['encoder']);rules=d['rules'];assert isinstance(rules,list) and len(rules)>0
        assert sum(r['kind']=='intercept' for r in rules)==1,'need exactly one intercept'
        complexity=0
        for r in rules:
            assert set(r)=={'kind','r','w'} and r['kind'] in ('intercept','skip','ltt')
            assert isinstance(r['r'],str) and isinstance(r['w'],dict)
            assert set(r['w']) <= {'tested_positive'}
            assert all(isinstance(w,(int,float)) and np.isfinite(w) for w in r['w'].values())
            if r['kind']=='intercept': assert r['r']=='True'
            if r['kind']=='skip': assert r['r'] in FEATURES
            complexity += 1+(sum(r['r'].count(op) for op in '&|^') if r['kind']=='ltt' else 0)
        result.append((RuleSet(rules,'binary',enc.class_names,complexity),enc))
    return result

def read_prediction(path):
    with open(path,newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==COLUMNS,'prediction header mismatch'
        rows=list(reader)
    assert rows,'empty predictions'
    assert all(set(r)==set(COLUMNS) and None not in r.values() for r in rows),'malformed CSV row'
    ids=[r['id'] for r in rows];assert len(set(ids))==len(ids) and all(ids),'duplicate/empty ids'
    certs=[r['certificate'] for r in rows if r['certificate']]
    assert len(certs)==1,'exactly one certificate required'
    card=json.loads(certs[0]);unpack(card)
    pred=np.asarray([[float(r[f'pred_{i}']) for i in range(5)] for r in rows])
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(),'probability must be finite and in [0,1]'
    return ids,pred,card

def write_prediction(path,ids,pred,card):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(COLUMNS)
        for j,id_ in enumerate(ids):w.writerow([id_,json.dumps(card,separators=(',',':')) if j==0 else '',*pred[j]])

def reconstruct(card,frame):
    models=unpack(card)
    return np.column_stack([predict_rules(r,frame[FEATURES],e) for r,e in models]),[r.complexity for r,e in models]

"""Frozen CSV readers and observable mesh/resource guards."""
import csv, math
NODE_COLUMNS=[f'x{i}' for i in range(11)]+[f'y{i}' for i in range(11)]
PRED_COLUMNS=['id']+NODE_COLUMNS+['train_s','inference_s','fem_solves']
INPUT_COLUMNS=['id','s1','s2']
LABEL_COLUMNS=INPUT_COLUMNS+['J_ref']
def read(path,columns):
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=columns:raise ValueError(f'Expected header {columns}')
        rows=list(r)
    if not rows:raise ValueError('Empty table')
    ids=[]
    for row in rows:
        if None in row or any(v is None for v in row.values()):raise ValueError('Invalid column count')
        ident=row['id']
        if not ident or str(int(ident))!=ident:raise ValueError('id must be a canonical integer')
        ids.append(ident)
        for c in columns[1:]:
            row[c]=float(row[c])
            if not math.isfinite(row[c]):raise ValueError(f'Nonfinite {c}')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate id')
    return rows

def matching(pred,inputs):
    if [r['id'] for r in pred]!=[r['id'] for r in inputs]:
        raise ValueError('Prediction ids/order must equal input ids/order')

def guards(rows):
    train_times={r['train_s'] for r in rows}
    if len(train_times)!=1:raise ValueError('Inconsistent training time')
    for r in rows:
        for axis in ['x','y']:
            n=[r[f'{axis}{i}'] for i in range(11)]
            if not (all(0<=v<=1 for v in n) and all(b>a for a,b in zip(n,n[1:])) and all(v in n for v in [0.,0.5,1.])):
                raise ValueError(f'Inadmissible {axis} mesh at id {r["id"]}')
        if r['train_s']<0:raise ValueError('Negative training time')
        if r['inference_s']<0:raise ValueError('Negative inference time')
        if r['fem_solves']!=0:raise ValueError('Parametric mesh prediction requires zero prediction-time FEM solves')

"""Strict CSV policy validator; no game evaluation and no candidate imports."""
import csv, math, sys
import numpy as np

COLUMNS=['id','instance','player','infoset','action','probability','updates','traversals','oracle']


def row_id(iid,p,j,a):return f'{iid}:{p}:{j}:{a}'


def read_prediction(path, data):
    ids=[str(x) for x in data['id']]
    expected={row_id(i,p,j,a):(i,int(p),int(j),int(a))
              for i in ids for p,j,_,a in data['schema']}
    rows={};groups={};counters={}
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLUMNS:raise ValueError('incorrect CSV header')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('incorrect column count')
            key=row['id']
            if key in rows:raise ValueError('duplicate row id')
            if key not in expected:raise ValueError('unexpected row id')
            iid,p,j,a=expected[key]
            if (row['instance'],row['player'],row['infoset'],row['action'])!=(iid,str(p),str(j),str(a)):
                raise ValueError('row metadata mismatch')
            values=[float(row[k]) for k in COLUMNS[5:]]
            if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite numeric value')
            prob,updates,traversals,oracle=values
            if not 0<=prob<=1:raise ValueError('probability outside [0,1]')
            if updates<0 or traversals<0 or not traversals.is_integer() or oracle not in (0,1):
                raise ValueError('invalid counter or oracle flag')
            counter=(updates,int(traversals),int(oracle))
            if iid in counters and counters[iid]!=counter:raise ValueError('inconsistent instance counters')
            counters[iid]=counter;rows[key]=row
            groups.setdefault((iid,p,j),[]).append(prob)
    if set(rows)!=set(expected):raise ValueError('missing policy rows')
    if any(abs(sum(v)-1)>1e-10 for v in groups.values()):raise ValueError('policy simplex violation')
    policies={i:{} for i in ids}
    for key,row in rows.items():
        iid,p,j,a=expected[key];policies[iid][(p,j,a)]=float(row['probability'])
    return policies,counters


def main(pred,input_path):
    with np.load(input_path,allow_pickle=False) as z:data={k:z[k] for k in ('id','schema')}
    read_prediction(pred,data)
    print('ok: policy rows, IDs, finite values, simplexes and counters')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))

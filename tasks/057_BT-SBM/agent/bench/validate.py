#!/usr/bin/env python3
import sys
from common import load, predictions

def main(pred,input_):
    x=load(input_)
    if type(x)!=dict or set(x)!={'datasets'} or type(x['datasets'])!=list or not x['datasets']: raise ValueError('invalid observed input')
    lengths=[]
    for ds in x['datasets']:
        if type(ds)!=dict or set(ds)!={'W'}: raise ValueError('observations must contain only W')
        W=ds['W']; n=len(W)
        if n<2 or any(type(r)!=list or len(r)!=n for r in W): raise ValueError('invalid W shape')
        if any(type(v)!=int or v<0 for r in W for v in r) or any(W[i][i]!=0 for i in range(n)): raise ValueError('invalid win counts')
        lengths.append(n)
    predictions(load(pred),lengths)
    print('ok:',len(lengths),'datasets')

if __name__=='__main__':
    try: main(*sys.argv[1:])
    except Exception as e: sys.exit(str(e))

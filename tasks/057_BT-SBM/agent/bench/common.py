import json, math
from collections import Counter

def load(path):
    def pairs(items):
        d = {}
        for k,v in items:
            if k in d: raise ValueError('duplicate JSON key: '+k)
            d[k]=v
        return d
    def invalid(x): raise ValueError('nonfinite JSON constant: '+x)
    with open(path) as f: return json.load(f, object_pairs_hook=pairs, parse_constant=invalid)

def finite(x): return type(x) in (int,float) and math.isfinite(x)

def predictions(p, lengths):
    if type(p)!=dict or set(p)!={'datasets'}: raise ValueError('prediction requires datasets only')
    ds=p['datasets']
    if type(ds)!=list or len(ds)!=len(lengths): raise ValueError('dataset count mismatch')
    for row,n in zip(ds,lengths):
        if type(row)!=dict or set(row)!={'partition','strength','fit_time_s'}: raise ValueError('invalid prediction keys')
        z,s=row['partition'],row['strength']
        if type(z)!=list or len(z)!=n or any(type(v)!=int or v<1 or v>2147483647 for v in z): raise ValueError('invalid partition')
        if set(z)!=set(range(1,len(set(z))+1)): raise ValueError('partition labels must be the consecutive integers 1..K')
        if type(s)!=list or len(s)!=n or any(not finite(v) or v<=0 for v in s): raise ValueError('invalid strengths')
        if not finite(row['fit_time_s']) or row['fit_time_s']<=0: raise ValueError('invalid fit time')
    return ds

def ari(a,b):
    # Exact installed fossil::adj.rand.index, including its original quirks:
    # table dimension = number of groups in group1, labels indexed 1..K,
    # entries outside that square are omitted, and the final value is absolute.
    # Do not silently replace this archived metric by standard signed ARI.
    choose=lambda n:n*(n-1)/2
    k=len(set(a)); n=len(a)
    cells=Counter((x,y) for x,y in zip(a,b) if 1<=x<=k and 1<=y<=k)
    ca=Counter(); cb=Counter()
    for (x,y),v in cells.items(): ca[x]+=v; cb[y]+=v
    pairs=sum(choose(v) for v in cells.values())
    sa=sum(choose(v) for v in ca.values()); sb=sum(choose(v) for v in cb.values())
    expected=sa*sb/choose(n)
    denominator=(sa+sb)/2-expected
    if denominator==0: raise ValueError('undefined fossil adjusted Rand index')
    return abs((pairs-expected)/denominator)

def rank(v):
    out=[0.]*len(v); order=sorted(range(len(v)),key=v.__getitem__); a=0
    while a<len(v):
        b=a+1
        while b<len(v) and v[order[b]]==v[order[a]]: b+=1
        for k in order[a:b]: out[k]=(a+b+1)/2
        a=b
    return out

def spearman(a,b):
    ra,rb=rank(a),rank(b); avg=(len(a)+1)/2
    da,db=[v-avg for v in ra],[v-avg for v in rb]
    denominator=math.sqrt(sum(v*v for v in da)*sum(v*v for v in db))
    if denominator==0: raise ValueError('undefined Spearman correlation')
    return sum(x*y for x,y in zip(da,db))/denominator

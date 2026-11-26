"""Frozen output schema and exact inclusive-endpoint scoring predicate."""
import hashlib,json,math,re,zipfile
HARD=(80,82,84,86);IDENTITIES=HARD+(95,100)
THRESHOLDS={'index_bytes':113716032.0,'offtarget_hsp_per_query':0.14,'identity_95_rate_pct':99.8910031488979,'identity_100_rate_pct':99.9}

def reject_constant(s):raise ValueError('nonfinite JSON constant: '+s)
def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key: '+k)
        d[k]=v
    return d

def loads(data):return json.loads(data,parse_constant=reject_constant,object_pairs_hook=unique_pairs)
def load(p):
    with open(p) as f:return loads(f.read())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def isint(x):return isinstance(x,int) and not isinstance(x,bool)
def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def check(pred,manifest,input_sha):
    assert set(pred)=={'version','input_sha256','query_ids','hsps','resources'},'prediction keys'
    assert pred['version']==1 and pred['input_sha256']==input_sha,'version or input digest'
    batches=manifest['batches']; expected=[q for b in batches for q in b['query_ids']]
    assert pred['query_ids']==expected and len(set(expected))==len(expected),'missing, duplicate, reordered or unknown query'
    byq={q:b for b in batches for q in b['query_ids']}
    assert isinstance(pred['hsps'],list),'HSP list'
    for h in pred['hsps']:
        assert isinstance(h,list) and len(h)==4,'HSP shape'
        q,s,a,z=h
        assert isinstance(q,str) and q in byq and isinstance(s,str) and s in byq[q]['lengths'],'unknown HSP query or subject'
        assert isint(a) and isint(z) and 1<=a<=byq[q]['lengths'][s] and 1<=z<=byq[q]['lengths'][s],'invalid HSP coordinates'
        assert abs(z-a)+1<=2000,'HSP span over 2000 bp'
    r=pred['resources']
    assert set(r)=={'cpus','seed','binary_sha256','source_sha256','compile_wall_s','indexed_groups','index_bytes','index_wall_s','search_wall_s','searched_batches'},'resource keys'
    assert isint(r['cpus']) and r['cpus']>0 and isint(r['seed']),'CPU/seed'
    assert r['indexed_groups']==manifest['indexed_groups'],'index coverage'
    assert r['searched_batches']==[b['name'] for b in batches],'search coverage'
    assert isint(r['index_bytes']) and r['index_bytes']>=0,'index bytes'
    for k in ('index_wall_s','search_wall_s','compile_wall_s'):assert number(r[k]) and r[k]>=0,'invalid time'
    for k in ('source_sha256','binary_sha256'):assert isinstance(r[k],str) and re.fullmatch('[0-9a-f]{64}',r[k]),'artifact digest'
    return pred

def positive(s,a,z,truth):return s==truth[0] and (truth[1]<=a<=truth[2] or truth[1]<=z<=truth[2])
def guard_checks(metrics,cpus,expected_cpus):
    guards={k:(v>=THRESHOLDS[k] if k.startswith('identity_') else v<=THRESHOLDS[k]) for k,v in metrics.items() if k in THRESHOLDS}
    guards['same_cpu_budget']=0<cpus<=expected_cpus
    return guards

def evaluate(pred,labels):
    check(pred,labels['manifest'],labels['input_sha256']);truth=labels['queries']
    assert set(truth)==set(pred['query_ids'])
    hits=set();off=0
    for q,s,a,z in pred['hsps']:
        if positive(s,a,z,truth[q]):hits.add(q)
        elif truth[q][3] in HARD:off+=1
    counts={i:sum(t[3]==i for t in truth.values()) for i in IDENTITIES}
    assert all(counts.values()),'all six identity strata required'
    rates={str(i):100*sum(truth[q][3]==i for q in hits)/counts[i] for i in IDENTITIES}
    raw=sum(rates[str(i)] for i in HARD)/4
    resources=pred['resources'];metrics={k:resources[k] for k in ('search_wall_s','index_wall_s','index_bytes')}
    metrics.update(offtarget_hsp_per_query=off/sum(counts[i] for i in HARD),identity_95_rate_pct=rates['95'],identity_100_rate_pct=rates['100'])
    guards=guard_checks(metrics,resources['cpus'],labels.get('expected_cpus',32))
    return {'alignment_rate':raw if all(guards.values()) else 0.0,'raw_alignment_rate':raw,'guards':guards,'constraint_metrics':metrics,'rate_by_identity':rates,'query_counts':{str(k):v for k,v in counts.items()},'guard_thresholds':THRESHOLDS}

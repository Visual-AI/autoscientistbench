"""Strict stdlib-only I/O shared by the frozen validator and scorer."""
import gzip,json
from pathlib import Path

def predictions(path, expected, edge_count):
    result={}
    with open(path) as f:
        if f.readline() != 'read_id\tedge\n':raise ValueError('expected read_id/edge TSV header')
        for line in f:
            cols=line.rstrip('\n').split('\t')
            if len(cols)!=2:raise ValueError('expected exactly two columns')
            rid,value=cols
            if rid not in expected:raise ValueError('unknown read id')
            if rid in result:raise ValueError('duplicate read id / multiple placement')
            edge=int(value)
            if str(edge)!=value or edge < -1 or edge>=edge_count:raise ValueError('invalid edge')
            result[rid]=edge
    if set(result)!=set(expected):raise ValueError('missing read ids')
    return result

def input_ids(path):
    path=Path(path);meta=json.loads((path/'manifest.json').read_text());ids=set();count=0
    with gzip.open(path/'reads.fq.gz','rt') as f:
        while True:
            h=f.readline()
            if not h:break
            seq=f.readline().rstrip('\n');plus=f.readline();qual=f.readline().rstrip('\n')
            if not h.startswith('@') or plus!='+\n' or len(seq)!=150 or len(qual)!=150:raise ValueError('invalid input fastq')
            rid=h[1:].rstrip('\n')
            if rid in ids:raise ValueError('duplicate input read')
            ids.add(rid);count+=1
    if count!=meta['n_reads']:raise ValueError('input count mismatch')
    return ids,meta

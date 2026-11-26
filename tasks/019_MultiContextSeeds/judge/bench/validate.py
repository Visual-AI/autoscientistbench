#!/usr/bin/env python3
"""Frozen schema and observed read coverage; no truth is consulted here."""
import gzip
import json
import math
from pathlib import Path
import re
import sys

ID=re.compile(r'read_[a-f0-9]{24}')
MAX_SPAN=300

def load_prediction(path):
    rows={}
    with open(path) as f:
        meta=json.loads(next(f))
        if set(meta)!={'schema','mapping_time_s_tool_reported','threads','seed'} or meta['schema']!=1:
            raise ValueError('invalid metadata schema')
        t=meta['mapping_time_s_tool_reported']
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0:
            raise ValueError('runtime must be finite and nonnegative')
        if type(meta['threads']) is not int or meta['threads']<1 or type(meta['seed']) is not int:
            raise ValueError('invalid threads/seed')
        if next(f)!='id\treference\tstart\tend\n':raise ValueError('invalid table header')
        for line in f:
            parts=line.rstrip('\n').split('\t')
            if len(parts)!=4:raise ValueError('invalid row width')
            key,ref,a,b=parts
            if not ID.fullmatch(key) or key in rows:raise ValueError('invalid or duplicate read ID')
            if not re.fullmatch(r'-?\d+',a) or not re.fullmatch(r'-?\d+',b):raise ValueError('noninteger interval')
            start,end=int(a),int(b)
            if ref=='*':
                if (start,end)!=(-1,-1):raise ValueError('invalid unmapped interval')
            elif start<0 or end<=start:raise ValueError('invalid mapped interval')
            elif end-start>MAX_SPAN:raise ValueError('mapped interval longer than 300 bases')
            rows[key]=(ref,start,end)
    if not rows:raise ValueError('empty predictions')
    return meta,rows

def validate(pred,inp):
    inp=Path(inp)
    cfg=json.loads((inp/'input.json').read_text())
    if cfg['protocol']=='full':
        if cfg['n_reads']!=1000000 or cfg['read_length']!=150 or cfg['reference']!='shared':raise ValueError('full protocol changed')
        expected_threads=12
    elif cfg['protocol']=='smoke':expected_threads=None
    else:raise ValueError('unknown protocol')
    meta,rows=load_prediction(pred)
    if expected_threads and meta['threads']>expected_threads:raise ValueError('full protocol permits at most 12 threads')
    lengths={k:int(v) for k,v in cfg['reference_lengths'].items()}
    for ref,start,end in rows.values():
        if ref!='*' and (ref not in lengths or end>lengths[ref]):raise ValueError('interval outside reference')
    ids=set()
    with gzip.open(inp/'reads.fastq.gz','rt') as f:
        for header in f:
            seq,plus,qual=next(f).strip(),next(f).strip(),next(f).strip()
            key=header[1:].strip()
            if not header.startswith('@') or not ID.fullmatch(key) or key in ids:raise ValueError('invalid observed read ID')
            if plus!='+' or len(seq)!=150 or len(qual)!=150:raise ValueError('invalid observed FASTQ')
            ids.add(key)
    if len(ids)!=cfg['n_reads'] or ids!=rows.keys():raise ValueError('prediction IDs must match every observed read exactly')
    return len(rows)

if __name__=='__main__':
    try:print('ok:',validate(*sys.argv[1:]),'reads')
    except Exception as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)

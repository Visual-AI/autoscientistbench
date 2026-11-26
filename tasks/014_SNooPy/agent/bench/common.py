"""Strict JSON/VCF contract shared by frozen validation and scoring."""
import json
import math
from pathlib import Path
import re

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result

def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def validate_prediction(pred_path, contigs):
    d = load_json(pred_path)
    if set(d) != {'version','contigs','vcf','runtime'} or d['version'] != 1:
        raise ValueError('wrong prediction schema/version')
    if d['contigs'] != contigs: raise ValueError('contig manifest mismatch')
    r = d['runtime']
    if set(r) != {'wall_s','cpus','exit_code','seed','caller_sha256'}:
        raise ValueError('runtime fields missing or extra')
    if type(r['wall_s']) not in (int,float) or not math.isfinite(r['wall_s']) or r['wall_s'] < 0:
        raise ValueError('invalid runtime')
    if any(type(r[k]) is not int for k in ('cpus','exit_code','seed')) or r['cpus'] < 1:
        raise ValueError('invalid CPU/exit/seed field')
    if not isinstance(r['caller_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',r['caller_sha256']):
        raise ValueError('invalid caller digest')
    vcf = d['vcf']
    if not isinstance(vcf,str) or not vcf.endswith('\n'): raise ValueError('VCF must end with newline')
    lines = vcf.splitlines()
    if not lines or lines[0] != '##fileformat=VCFv4.2': raise ValueError('VCF version header missing')
    header = False; declared = {}; seen = set(); rows = 0
    for line in lines[1:]:
        if line.startswith('##'):
            if header: raise ValueError('metadata after VCF columns')
            if line.startswith('##contig='):
                m = re.fullmatch(r'##contig=<ID=([^,>]+),length=(\d+)>',line)
                if not m or m[1] in declared: raise ValueError('invalid or duplicate contig header')
                declared[m[1]] = int(m[2])
            continue
        if line.startswith('#'):
            if header or line != '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO':
                raise ValueError('invalid VCF columns')
            header = True; continue
        if not header: raise ValueError('VCF columns missing')
        f = line.split('\t')
        if len(f) != 8: raise ValueError('VCF requires eight columns')
        c,pos,ref,alt = f[0],int(f[1]),f[3],f[4]
        if c not in contigs or not 1 <= pos <= contigs[c]: raise ValueError('variant outside contigs')
        if not re.fullmatch(r'[ACGTNacgtn.*]+',ref) or not all(re.fullmatch(r'[ACGTNacgtn.*]+',x) for x in alt.split(',')):
            raise ValueError('invalid allele')
        if pos+len(ref)-1 > contigs[c]: raise ValueError('allele extends beyond contig')
        if f[5] != '.' and (not math.isfinite(float(f[5])) or float(f[5]) < 0): raise ValueError('invalid QUAL')
        key=(c,pos,ref,alt)
        if key in seen: raise ValueError('duplicate variant')
        seen.add(key); rows += 1
    if not header or declared != contigs: raise ValueError('VCF contig headers incomplete')
    return d, rows

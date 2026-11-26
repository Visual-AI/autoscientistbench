#!/usr/bin/env python3
"""Exact authors SNP position metric on pre-existing separated truth calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from authors_metric import count_number_of_missing_variants
from common import load_json, validate_prediction

def score(prediction, labels):
    labels = Path(labels).resolve()
    spec = load_json(labels/'manifest.json')
    try:
        d, _ = validate_prediction(prediction,spec['contigs'])
    except Exception as exc:
        return {'snp_f1':0.0,'guard_passed':False,'reason':'malformed prediction: '+str(exc)}
    r = d['runtime']
    if r['cpus'] > 32 or r['exit_code'] != 0:
        return {'snp_f1':0.0,'guard_passed':False,'reason':'32 CPU / successful calling guard'}
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='snp-score-') as tmp:
        try:
            os.chdir(tmp)
            Path('prediction.vcf').write_text(d['vcf'])
            with contextlib.redirect_stdout(io.StringIO()):
                result = count_number_of_missing_variants(str(labels/'few.vcf'),str(labels/'many.vcf'),'prediction.vcf')[0]
        finally:
            os.chdir(old)
    p, recall = result['precision'],result['recall']
    f1 = 2*p*recall/(p+recall) if p+recall else 0.0
    return {'snp_f1':f1,'precision':p,'recall':recall,'true_variants':result['true_variants'],
            'called_variants':result['called_variants'],'guard_passed':True}

if __name__ == '__main__':
    print(json.dumps(score(sys.argv[1],sys.argv[2]),allow_nan=False))

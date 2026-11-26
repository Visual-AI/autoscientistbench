#!/usr/bin/env python3
"""Exact authors' unit-edge error, with the manifest's coverage guards."""
import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_common import predictions

OVERALL_FLOOR={'test':0.881,'val':0.858}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred_path,labels,overall_floor=OVERALL_FLOOR['test']):
    mapping=labels['read_query'];errors=labels['edge_errors'];root=labels['root_edge']
    counts=collections.Counter(mapping.values());worst=sum(counts[q]*errors[q][root] for q in counts)/len(mapping)
    try:pred=predictions(pred_path,mapping,labels['edge_count'])
    except Exception as e:return {'mean_edge_error':worst,'guards_passed':False,'failure':str(e),'worst':worst}
    nonroot=collections.Counter();total_error=0
    for rid,edge in pred.items():
        if edge in (-1,root):continue
        q=mapping[rid];nonroot[q]+=1;total_error+=errors[q][edge]
    placed=sum(nonroot.values());overall=placed/len(mapping)
    rates={q:nonroot[q]/counts[q] for q in counts}
    # Equality is admitted at each threshold; floats tolerate only representation error.
    failed=[q for q in counts if rates[q]+1e-12<labels['baseline_rates'][q]-0.10]
    ok=overall+1e-12>=overall_floor and not failed and placed>0
    return {'mean_edge_error':total_error/placed if ok else worst,'guards_passed':ok,'overall_placement_rate':overall,'per_query_rates':rates,'failed_query_guards':failed,'nonroot_placements':placed,'n_reads':len(mapping),'worst':worst}
if __name__=='__main__':
    print(json.dumps(score(sys.argv[1],json.load(open(sys.argv[2])),OVERALL_FLOOR[split_of(sys.argv[2])]),allow_nan=False))

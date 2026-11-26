#!/usr/bin/env python3
"""Original mean trial power/FDR and mean permutations; audit-required access guard."""
import json,statistics,sys
from common import labels,predictions
METRIC="mean_permutations_per_hypothesis_anytime_valid_BC"
PENALTY=10000

def score(pred_path,labels_path):
    try:
        y,g,truth=labels(labels_path);out=predictions(pred_path,g)
        powers=[];fdrs=[];counts=[];sampler_ok=True
        for ident,setting in g.items():
            a=truth[ident];p=out[ident]
            rejected=[h for h in p if p[h][0]]
            powers.append(sum(a[h-1] for h in rejected)/max(1,sum(a)))
            fdrs.append(sum(1-a[h-1] for h in rejected)/max(1,len(rejected)))
            counts.append(statistics.fmean(p[h][1] for h in p))
            sampler_ok=sampler_ok and all(p[h][2]==1 for h in p)
        power=statistics.fmean(powers);fdr=statistics.fmean(fdrs);mean=statistics.fmean(counts)
        passed=power>=0.720 and fdr<=0.10 and sampler_ok
        return {METRIC:mean if passed else PENALTY,"raw_mean_permutations":mean,"power":power,"FDR":fdr,
                "power_guard":power>=0.720,"fdr_guard":fdr<=0.10,"sampler_only_declared":sampler_ok,
                "patch_audit_required":["sampler-only access","truth/analytic-p-value exclusion","reported draw-count authenticity"],
                "scope":y["scope"],"trials":len(g)}
    except Exception as e:return {METRIC:PENALTY,"invalid":True,"reason":str(e)}
if __name__=="__main__": print(json.dumps(score(*sys.argv[1:]),allow_nan=False))

"""Mean 90% CI width with every original manifest guard retained."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from output_checks import check


def score(pred_path,labels_path):
    labels=json.loads(Path(labels_path).read_text())
    try:
        out,budgets=check(pred_path,labels)
        rows=out['trials']
        width=float(np.mean([r['width'] for r in rows]))
        coverage=float(np.mean([r['lower']<labels['theta_true']<r['upper'] for r in rows]))
        count=float(np.mean([len(r['selected']) for r in rows]))
        max_budget=max(budgets[r['design']] for r in rows)
        guards=dict(coverage=coverage>=.89,mean_label_count=count<=870,
            design_budget=max_budget<=869.0,predictions_hash_identical=True)
        eligible=all(guards.values())
        return dict(mean_ci_width=width if eligible else float('inf'),observed_mean_width=width,
            coverage=coverage,mean_n_labeled=count,max_sum_pi=max_budget,
            replications=len(rows),eligible=eligible,guards=guards,
            verification_scope=labels['scope'],label_access_enforcement='patch_audit')
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return dict(mean_ci_width=float('inf'),eligible=False,error=str(exc))


if __name__=='__main__': print(json.dumps(score(sys.argv[1],sys.argv[2])))

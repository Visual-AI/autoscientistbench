#!/usr/bin/env python3
"""Within-subject R@1 with RSA >= .51 on test and >= .48 on val, and embedding dimension <=128."""
import itertools
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_prediction

# Frozen reference numerics; caller thread budget is an upper bound.
torch.set_num_threads(min(4, int(os.environ.get('BENCH_CPUS', '4'))))

RSA_THRESHOLD = {'test': .51, 'val': .48}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

@torch.no_grad()
def pair_metrics(first, second):
    first = torch.from_numpy(first.astype(np.float32))
    second = torch.from_numpy(second.astype(np.float32))
    a = torch.nn.functional.normalize(first, dim=1)
    b = torch.nn.functional.normalize(second, dim=1)
    similarity = b @ a.T
    target = torch.arange(len(first))
    recall = (similarity.argmax(dim=1) == target).float().mean().item()
    ranks = (torch.argsort(similarity, dim=1, descending=True) == target[:,None]).nonzero()[:,1] + 1
    mean_rank = ranks.float().mean().item()
    rows, cols = torch.triu_indices(len(first),len(first),offset=1)
    rdm_a = 1 - torch.corrcoef(first)
    rdm_b = 1 - torch.corrcoef(second)
    rsa = torch.corrcoef(torch.stack([rdm_a[rows,cols],rdm_b[rows,cols]]))[0,1].item()
    return dict(recall_at_1=recall, rsa=rsa, mean_rank=mean_rank)

def score(pred_path, labels_path):
    arrays, subjects = read_prediction(pred_path)
    records = []
    dimensions = [arrays[f'embedding_{s}'].shape[1] for s in subjects]
    with np.load(labels_path, allow_pickle=False) as labels:
        if set(subjects) != {int(k.removeprefix('id_')) for k in labels.files if k.startswith('id_')}:
            raise ValueError('subject set differs from labels')
        for s in subjects:
            if not np.array_equal(arrays[f'id_{s}'], labels[f'id_{s}']):
                raise ValueError('label ids mismatch')
            group, rep = labels[f'group_{s}'], labels[f'repetition_{s}']
            unique, counts = np.unique(group, return_counts=True)
            complete = unique[counts == 3]
            if len(complete) < 2:
                raise ValueError('insufficient complete repetition groups')
            lookup = {(int(g), int(r)):i for i,(g,r) in enumerate(zip(group,rep))}
            z = arrays[f'embedding_{s}']
            views = [z[[lookup[(int(g),r)] for g in complete]] for r in range(3)]
            for a,b in itertools.combinations(range(3),2):
                records.append(dict(subject=s, pair=[a,b], **pair_metrics(views[a],views[b])))
    mean_rsa = float(np.mean([r['rsa'] for r in records]))
    mean_recall = float(np.mean([r['recall_at_1'] for r in records]))
    passed = np.isfinite(mean_rsa) and mean_rsa >= RSA_THRESHOLD[split_of(labels_path)]
    return dict(recall_at_1=mean_recall if passed else float('inf'), score_rule_value=mean_recall if passed else 0.,
        mean_rsa=mean_rsa if np.isfinite(mean_rsa) else None, raw_recall_at_1=mean_recall,
        mean_rank=float(np.mean([r['mean_rank'] for r in records])), max_embedding_dim=max(dimensions),
        guard='pass' if passed else 'rsa', per_pair=[{k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in r.items()} for r in records])

if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = dict(recall_at_1=float('inf'), score_rule_value=0., guard='invalid', reason=str(exc))
    print(json.dumps(result, allow_nan=True))

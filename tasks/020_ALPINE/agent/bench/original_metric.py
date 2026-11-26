#!/usr/bin/env python3
"""Headline metric of the paper's main experiment (Results 3.1, Tables S1-S4).

The published ALPINE benchmark (Zenodo doi:10.5281/zenodo.20260865) is four
simulated pools - PacBio HiFi / ONT x TRAC / TRBC - of 60,000 reads each, whose
read names carry the ground-truth outcome group (see tools/pool_ground_truth.py).
ALPINE's per-sample output `readname_<sample>.txt` pairs every read with the
category it assigned to it, so precision / recall / F1 can be computed exactly
as in the authors' implementation
(ALPINE-manuscript-analysis/simulation_benchmark/check_accuracy_alpine.py):

    recall(group)   = reads of the group classified as the group's expected
                      class / reads of the group
    precision(cls)  = reads classified as `cls` that come from a group whose
                      expected class is `cls` / all reads classified as `cls`
    F1              = harmonic mean of the two

Transgene integration categories are HDR, ITR (Non-HDR-with-ITR), Truncated HDR
(Non-HDR-without-ITR) and the two cross-target counterparts.  The paper's
headline number is the mean integration F1 over the two target sites:
    PacBio HiFi 98.95 %  (98.45 TRAC + 99.44 TRBC)  Results 3.1, Tables S1-S2
    ONT         99.63 %  (99.62 TRAC + 99.63 TRBC)                  Tables S3-S4
The paper also reports, per pool, the overall read-level F1 (its "Total" row);
reads assigned to a class that is not the target class of any simulated group
(e.g. Unclassified, or the mirrored cross-target class) are reported in the
paper's "Unclassified/Other" row and we reproduce that bookkeeping here.

Inputs are only the classification outputs produced by run_train.sh.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAPER_SOURCE = "Results 3.1 (p. 4), Tables S1-S4 and Fig. S6, Bioinformatics btag528"

# Supplementary Tables S1-S4: (paper row label, reads, ALPINE class, TP, FP, FN, precision, recall, F1)
PAPER = {
    "hifi_trac": {
        "table": "Table S1 (Simulated PacBio HiFi dataset benchmark - TRAC site)",
        "rows": [
            ("HDR",              5000,  "HDR-CD19",                    4658,    0,  342, 100.00, 93.16, 96.46),
            ("ITR",             10000,  "Non-HDR-with-ITR-CD19",       9999,    0,    1, 100.00, 99.99, 99.99),
            ("Truncated HDR",    5000,  "Non-HDR-without-ITR-CD19",    4900,  343,  100,  93.46, 98.00, 95.68),
            ("Cross-Target-HDR", 5000,  "HDR-CD22",                    4860,    0,  140, 100.00, 97.20, 98.58),
            ("Cross-Target-ITR", 5000,  "Non-HDR-with-ITR-CD22",       5000,    0,    0, 100.00, 100.00, 100.00),
            ("Wild type",        5000,  "Unmodified",                  4472,  465,  528,  90.58, 89.44, 90.01),
            ("SNP",              5000,  "Unmodified-with-SNP",         4460,   21,  540,  99.53, 89.20, 94.08),
            ("large DEL",        5000,  "DEL-large",                   4949,   87,   51,  98.27, 98.98, 98.62),
            ("small DEL",        5000,  "DEL-small",                   4910,  565,   90,  89.68, 98.20, 93.75),
            ("large INS",        5000,  "INS-large",                   4774,  183,  226,  96.31, 95.48, 95.89),
            ("small INS",        5000,  "INS-small",                   4667,  543,  333,  89.58, 93.34, 91.42),
            ("-",                   0,  "Unclassified/Other",             0,  144,    0,   0.00,  0.00,  0.00),
        ],
        "integration": (29417, 343, 583, 98.85, 98.06, 98.45),
        "total": (57649, 2351, 2351, 96.08),
    },
    "hifi_trbc": {
        "table": "Table S2 (Simulated PacBio HiFi dataset benchmark - TRBC site)",
        "rows": [
            ("HDR",              5000,  "HDR-CD22",                    4897,    0,  103, 100.00, 97.94, 98.96),
            ("ITR",             10000,  "Non-HDR-with-ITR-CD22",       9999,    0,    1, 100.00, 99.99, 99.99),
            ("Truncated HDR",    5000,  "Non-HDR-without-ITR-CD22",    5000,  104,    0,  97.96, 100.00, 98.97),
            ("Cross-Target-HDR", 5000,  "HDR-CD19",                    4872,    0,  128, 100.00, 97.44, 98.70),
            ("Cross-Target-ITR", 5000,  "Non-HDR-with-ITR-CD19",       5000,    0,    0, 100.00, 100.00, 100.00),
            ("Wild type",        5000,  "Unmodified",                  4251,  562,  749,  88.32, 85.02, 86.64),
            ("SNP",              5000,  "Unmodified-with-SNP",         4193,   31,  807,  99.27, 83.86, 90.92),
            ("large DEL",        5000,  "DEL-large",                   4930,   96,   70,  98.09, 98.60, 98.34),
            ("small DEL",        5000,  "DEL-small",                   4899, 1175,  101,  80.66, 97.98, 88.48),
            ("large INS",        5000,  "INS-large",                   4710,   42,  290,  99.12, 94.20, 96.60),
            ("small INS",        5000,  "INS-small",                   4633,  470,  367,  90.79, 92.66, 91.72),
            ("-",                   0,  "Unclassified/Other",             0,  136,    0,   0.00,  0.00,  0.00),
        ],
        "integration": (29768, 104, 232, 99.65, 99.23, 99.44),
        "total": (57384, 2616, 2616, 95.64),
    },
    "ont_trac": {
        "table": "Table S3 (Simulated ONT dataset benchmark - TRAC site)",
        "rows": [
            ("HDR",              5000,  "HDR-CD19",                    4999,   64,    1,  98.74, 99.98, 99.35),
            ("ITR",             10000,  "Non-HDR-with-ITR-CD19",       9896,    0,  104, 100.00, 98.96, 99.48),
            ("Truncated HDR",    5000,  "Non-HDR-without-ITR-CD19",    4993,   41,    7,  99.19, 99.86, 99.52),
            ("Cross-Target-HDR", 5000,  "HDR-CD22",                    4992,    0,    8, 100.00, 99.84, 99.92),
            ("Cross-Target-ITR", 5000,  "Non-HDR-with-ITR-CD22",       5000,    0,    0, 100.00, 100.00, 100.00),
            ("Wild type",        5000,  "Unmodified",                  1111,  129, 3889,  89.60, 22.22, 35.61),
            ("SNP",              5000,  "Unmodified-with-SNP",         2320, 1318, 2680,  63.77, 46.40, 53.72),
            ("large DEL",        5000,  "DEL-large",                   4750,   97,  250,  98.00, 95.00, 96.48),
            ("small DEL",        5000,  "DEL-small",                   4892, 4011,  108,  54.95, 97.84, 70.37),
            ("large INS",        5000,  "INS-large",                   4720,   76,  280,  98.42, 94.40, 96.37),
            ("small INS",        5000,  "INS-small",                   4624, 1896,  376,  70.92, 92.48, 80.28),
            ("-",                   0,  "Unclassified/Other",             0,   71,    0,   0.00,  0.00,  0.00),
        ],
        "integration": (29880, 105, 120, 99.65, 99.60, 99.62),
        "total": (52297, 7703, 7703, 87.16),
    },
    "ont_trbc": {
        "table": "Table S4 (Simulated ONT dataset benchmark - TRBC site)",
        "rows": [
            ("HDR",              5000,  "HDR-CD22",                    4999,   79,    1,  98.44, 99.98, 99.21),
            ("ITR",             10000,  "Non-HDR-with-ITR-CD22",       9893,    0,  107, 100.00, 98.93, 99.46),
            ("Truncated HDR",    5000,  "Non-HDR-without-ITR-CD22",    4998,   29,    2,  99.42, 99.96, 99.69),
            ("Cross-Target-HDR", 5000,  "HDR-CD19",                    4995,    0,    5, 100.00, 99.90, 99.95),
            ("Cross-Target-ITR", 5000,  "Non-HDR-with-ITR-CD19",       5000,    0,    0, 100.00, 100.00, 100.00),
            ("Wild type",        5000,  "Unmodified",                  1126,  178, 3874,  86.35, 22.52, 35.72),
            ("SNP",              5000,  "Unmodified-with-SNP",         2218, 1260, 2782,  63.77, 44.36, 52.32),
            ("large DEL",        5000,  "DEL-large",                   4746,   87,  254,  98.20, 94.92, 96.53),
            ("small DEL",        5000,  "DEL-small",                   4890, 4480,  110,  52.19, 97.80, 68.06),
            ("large INS",        5000,  "INS-large",                   4668,   51,  332,  98.92, 93.36, 96.06),
            ("small INS",        5000,  "INS-small",                   4610, 1619,  390,  74.01, 92.20, 82.11),
            ("-",                   0,  "Unclassified/Other",             0,   74,    0,   0.00,  0.00,  0.00),
        ],
        "integration": (29885, 108, 115, 99.64, 99.62, 99.63),
        "total": (52143, 7857, 7857, 86.91),
    },
}

INTEGRATION_GROUPS = ("hdr", "itr", "truncated_hdr", "cross_hdr", "cross_itr")


def r2(x):
    """round-half-up to 2 decimals (the paper rounds 98.945 -> 98.95)"""
    return math.floor(x * 100 + 0.5) / 100


def read_class_map(path):
    """class_map.tsv -> {pool: [(group, class, expected_reads, row_label), ...]}"""
    out = collections.OrderedDict()
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        pool, group, klass, expected, order, label = line.rstrip("\n").split("\t")
        out.setdefault(pool, []).append((int(order), group, klass, int(expected), label))
    for pool in out:
        out[pool].sort()
    return collections.OrderedDict(
        (pool, [(g, k, n, label) for _o, g, k, n, label in rows]) for pool, rows in out.items())


def read_pairs(path):
    """readname_<sample>.txt -> ((group, class) -> reads, total reads)."""
    pairs = collections.Counter()
    total = 0
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, klass = line.split("\t")[:2]
        group = name.split(".")[0].replace("itr_oneside", "itr")
        pairs[(group, klass)] += 1
        total += 1
    return pairs, total


def f1_of(tp, fp, fn):
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    rec = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate_pool(pool, rows, pairs, total_reads):
    by_group = collections.defaultdict(int)
    for (g, _c), v in pairs.items():
        by_group[g] += v
    fp_by_class = collections.Counter()
    for (g, c), v in pairs.items():
        expected = dict((r[0], r[1]) for r in rows).get(g)
        if expected != c:
            fp_by_class[c] += v

    table_rows, tot_tp = [], 0
    integration = [0, 0, 0]
    integration_classes = set()
    for group, klass, _n, label in rows:
        tp = pairs.get((group, klass), 0)
        n = by_group[group]
        fn = n - tp
        fp = fp_by_class[klass]
        prec, rec, f1 = f1_of(tp, fp, fn)
        table_rows.append(dict(row_label=label, group=group, klass=klass, reads=n, tp=tp, fp=fp,
                               fn=fn, precision=round(prec, 2), recall=round(rec, 2), f1=round(f1, 2)))
        tot_tp += tp
        if any(group.endswith(s) for s in ("_cd19_hdr", "_cd22_hdr")) or \
           any(group.endswith(s) for s in ("_cd19_itr", "_cd22_itr")) or \
           group.endswith(("_cd19_truncated_hdr", "_cd22_truncated_hdr")):
            integration[0] += tp
            integration[1] += fp
            integration[2] += fn
            integration_classes.add(klass)
    # unclassified / other: reads whose class is not the target class of any group
    known = {r[1] for r in rows}
    extra = sum(v for c, v in fp_by_class.items() if c not in known)
    iprec, irec, if1 = f1_of(*integration)
    tprec, trec, tf1 = f1_of(tot_tp, total_reads - tot_tp, total_reads - tot_tp)
    return dict(
        pools_rows=table_rows,
        unclassified_other_fp=extra,
        integration=dict(tp=integration[0], fp=integration[1], fn=integration[2],
                         precision=round(iprec, 2), recall=round(irec, 2), f1=round(if1, 2)),
        total=dict(reads=total_reads, tp=tot_tp, fp=total_reads - tot_tp, fn=total_reads - tot_tp,
                   precision=round(tprec, 2), recall=round(trec, 2), f1=round(tf1, 2)),
    )


def print_comparison(pool, res):
    paper = PAPER[pool]
    print(f"\n=== {paper['table']} ===")
    print(f"{'row':<18} {'class':<26} {'TP ours':>8} {'TP paper':>9} {'FP ours':>8} {'FP paper':>9} "
          f"{'FN ours':>8} {'FN paper':>9} {'F1 ours':>8} {'F1 paper':>9}")
    for ours, prow in zip(res["pools_rows"], paper["rows"]):
        label, _reads, klass, ptp, pfp, pfn, _pp, _pr, pf1 = prow
        mark = "" if (ours["tp"] == ptp and ours["fp"] == pfp and ours["fn"] == pfn) else "  <-- differs"
        if label == "-":
            ours_fp = res["unclassified_other_fp"]
            print(f"{'Unclassified/Other':<18} {klass:<26} {0:>8} {0:>9} {ours_fp:>8} {pfp:>9} "
                  f"{0:>8} {0:>9} {0.0:>8.2f} {0.0:>9.2f}"
                  f"{'' if ours_fp == pfp else '  <-- differs'}")
        else:
            print(f"{label:<18} {klass:<26} {ours['tp']:>8} {ptp:>9} {ours['fp']:>8} {pfp:>9} "
                  f"{ours['fn']:>8} {pfn:>9} {ours['f1']:>8.2f} {pf1:>9.2f}{mark}")
    it, pt = res["integration"], paper["integration"]
    print(f"{'Total integration':<18} {'':<26} {it['tp']:>8} {pt[0]:>9} {it['fp']:>8} {pt[1]:>9} "
          f"{it['fn']:>8} {pt[2]:>9} {it['f1']:>8.2f} {pt[5]:>9.2f}")
    tt, ptot = res["total"], paper["total"]
    print(f"{'Total (all reads)':<18} {'':<26} {tt['tp']:>8} {ptot[0]:>9} {tt['fp']:>8} {ptot[1]:>9} "
          f"{tt['fn']:>8} {ptot[2]:>9} {tt['f1']:>8.2f} {ptot[3]:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", default=os.path.join(ROOT, "results", "per_sample"))
    ap.add_argument("--class-map", default=os.path.join(ROOT, "data", "bench", "class_map.tsv"))
    ap.add_argument("--metrics-json", default=os.path.join(ROOT, "results", "metrics.json"))
    ap.add_argument("--tables-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    class_map = read_class_map(args.class_map)
    results, missing = {}, []
    for pool in PAPER:
        hits = sorted(glob.glob(os.path.join(args.per_sample, pool, "**", f"readname_{pool}.txt"),
                                recursive=True))
        if not hits:
            missing.append(pool)
            continue
        pairs, total = read_pairs(hits[0])
        res = evaluate_pool(pool, class_map[pool], pairs, total)
        res["readname_file"] = hits[0]
        results[pool] = res
        print_comparison(pool, res)
    if missing:
        print(f"\n[eval] missing classification output for: {', '.join(missing)}", file=sys.stderr)
        return 1

    hifi = [results[p]["integration"]["f1"] for p in ("hifi_trac", "hifi_trbc")]
    ont = [results[p]["integration"]["f1"] for p in ("ont_trac", "ont_trbc")]
    hifi_mean, ont_mean = r2(sum(hifi) / 2), r2(sum(ont) / 2)
    paper_hifi, paper_ont = 98.95, 99.63
    print("\n=== Headline metric (Results 3.1) ===")
    print("transgene-integration F1, mean over the two target sites (TRAC, TRBC):")
    print(f"  PacBio HiFi : reproduced {hifi_mean:.2f}  (paper {paper_hifi:.2f})  "
          f"relative gap {100*(hifi_mean-paper_hifi)/paper_hifi:+.2f}%   [TRAC {hifi[0]:.2f}, TRBC {hifi[1]:.2f}]")
    print(f"  ONT         : reproduced {ont_mean:.2f}  (paper {paper_ont:.2f})  "
          f"relative gap {100*(ont_mean-paper_ont)/paper_ont:+.2f}%   [TRAC {ont[0]:.2f}, TRBC {ont[1]:.2f}]")

    for pool in results:
        out_tsv = os.path.join(args.tables_dir, f"benchmark_{pool}.tsv")
        with open(out_tsv, "w") as fh:
            fh.write("#reproduced ALPINE benchmark table for pool %s\n" % pool)
            fh.write("#row\tclass\treads\tTP\tFP\tFN\tprecision\trecall\tF1\t"
                     "paper_TP\tpaper_FP\tpaper_FN\tpaper_F1\n")
            for ours, prow in zip(results[pool]["pools_rows"], PAPER[pool]["rows"]):
                label, _r, klass, ptp, pfp, pfn, _p, _rr, pf1 = prow
                if label == "-":
                    fh.write(f"Unclassified/Other\t{klass}\t0\t0\t{results[pool]['unclassified_other_fp']}"
                             f"\t0\t0.00\t0.00\t0.00\t0\t{pfp}\t0\t0.00\n")
                else:
                    fh.write(f"{ours['row_label']}\t{ours['klass']}\t{ours['reads']}\t{ours['tp']}\t"
                             f"{ours['fp']}\t{ours['fn']}\t{ours['precision']:.2f}\t{ours['recall']:.2f}\t"
                             f"{ours['f1']:.2f}\t{ptp}\t{pfp}\t{pfn}\t{pf1:.2f}\n")
            it, pt = results[pool]["integration"], PAPER[pool]["integration"]
            fh.write(f"Total integration\t-\t30000\t{it['tp']}\t{it['fp']}\t{it['fn']}\t{it['precision']:.2f}\t"
                     f"{it['recall']:.2f}\t{it['f1']:.2f}\t{pt[0]}\t{pt[1]}\t{pt[2]}\t{pt[5]:.2f}\n")
            tt, ptot = results[pool]["total"], PAPER[pool]["total"]
            fh.write(f"Total\t-\t{tt['reads']}\t{tt['tp']}\t{tt['fp']}\t{tt['fn']}\t{tt['precision']:.2f}\t"
                     f"{tt['recall']:.2f}\t{tt['f1']:.2f}\t{ptot[0]}\t{ptot[1]}\t{ptot[2]}\t{ptot[3]:.2f}\n")
        print(f"[eval] wrote {out_tsv}")

    metrics = {
        "primary_metric": {
            "name": "Mean transgene-integration F1 over the two target sites (TRAC + TRBC), "
                    "simulated PacBio HiFi dual-target benchmark",
            "higher_is_better": True,
            "setting": "ALPINE CWL classification of 60,000 simulated HiFi reads at each of TRAC and TRBC "
                       "(11 ground-truth outcome groups, 5 integration groups); F1 from TP/FP/FN against "
                       "the simulation labels, averaged over the two target sites",
            "paper_value": paper_hifi,
            "reproduced_value": hifi_mean,
            "relative_gap": round((hifi_mean - paper_hifi) / paper_hifi, 6),
            "paper_source": PAPER_SOURCE,
        },
        "secondary_metrics": [
            {"name": "Mean transgene-integration F1 over the two target sites, simulated ONT benchmark",
             "higher_is_better": True, "paper_value": paper_ont, "reproduced_value": ont_mean,
             "relative_gap": round((ont_mean - paper_ont) / paper_ont, 6)},
        ] + [
            {"name": f"Overall read-level F1, {pool} pool (paper 'Total' row)",
             "higher_is_better": True, "paper_value": PAPER[pool]["total"][3],
             "reproduced_value": results[pool]["total"]["f1"],
             "relative_gap": round((results[pool]["total"]["f1"] - PAPER[pool]["total"][3]) /
                                   PAPER[pool]["total"][3], 6)}
            for pool in PAPER
        ] + [
            {"name": f"Transgene-integration F1, {pool} pool",
             "higher_is_better": True, "paper_value": PAPER[pool]["integration"][5],
             "reproduced_value": results[pool]["integration"]["f1"],
             "relative_gap": round((results[pool]["integration"]["f1"] - PAPER[pool]["integration"][5]) /
                                   PAPER[pool]["integration"][5], 6)}
            for pool in PAPER
        ],
        "pools": {p: {k: v for k, v in results[p].items()} for p in results},
        "paper_source": PAPER_SOURCE,
    }
    with open(args.metrics_json, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[eval] wrote {args.metrics_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

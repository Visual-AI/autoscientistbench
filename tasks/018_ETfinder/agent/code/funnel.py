#!/usr/bin/env python3
"""Funnel counting for the Bio-18 main experiment (ETfinder, paper Fig. 2A).

Re-implements the three counting rules of the paper's R. sphaeroides funnel that
the released CLI does not expose, on the released RecTdb:

  1. SSB-C7 restriction  : every (RecT, host) record whose host SSB C-terminus
                           equals the query motif (paper: "RecT homologs", 638)
  2. cognate RecE-RecT   : records carrying an exonuclease/RecE partner in their
                           stored genomic neighbourhood, using the authors' own
                           Pfam list from templates/result.html
                           ('PF12684.10_DUF3799', 'PF12705.10_PDDEXK_1',
                            'PF09588.13_YqaJ')          (paper: 198 pairs)
  3. host redundancy     : 16S rDNA of the surviving hosts (RefSeq
                           *_rna_from_genomic.fna.gz) clustered with cd-hit-est
                           (paper: 91 unique RecET homolog pairs)

Usage: funnel.py <rectdb.json> <assembly_summary_refseq.txt> <outdir> <16S_cache_dir>
"""
import collections
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MOTIF = os.environ.get("ET_MOTIF", "LDDEIPF")
EXO = ["PF12684.10_DUF3799", "PF12705.10_PDDEXK_1", "PF09588.13_YqaJ"]  # authors' filter
LEN_MIN, LEN_MAX = 200, 700
THREADS = os.environ.get("ET_THREADS", "12")


def ftp_paths(summary: Path):
    """accession -> (ftp_path, asm_name)"""
    out = {}
    with open(summary, errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 20:
                continue
            out[f[0]] = (f[19], f[15])
    return out


def download_16s(gcf, ftp, asm, cache: Path):
    """Return the 16S rDNA sequence of an assembly (longest 16S record), cached."""
    gz = cache / f"{gcf}_rna_from_genomic.fna.gz"
    if not gz.exists() or gz.stat().st_size < 200:
        return None  # offline snapshot: preserve existing missing-16S cases
    best = None
    head, buf = None, []
    with gzip.open(gz, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if head and "16S ribosomal RNA" in head:
                    seq = "".join(buf)
                    if best is None or len(seq) > len(best):
                        best = seq
                head, buf = line, []
            else:
                buf.append(line.strip())
    if head and "16S ribosomal RNA" in head:
        seq = "".join(buf)
        if best is None or len(seq) > len(best):
            best = seq
    return best


def cluster16s(seqs, outdir, ident):
    """cd-hit-est clustering of the 16S sequences; returns cluster key per record."""
    fa = outdir / f"host16S_{int(ident*100)}.fasta"
    with open(fa, "w") as fh:
        for key, seq in seqs.items():
            fh.write(f">{key}\n{seq}\n")
    pref = outdir / f"host16S_{int(ident*100)}_cdhit"
    subprocess.run(["cd-hit-est", "-i", str(fa), "-o", str(pref), "-c", str(ident),
                    "-aL", "0.9", "-T", str(THREADS), "-M", "4000"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key2cl, cid = {}, None
    with open(str(pref) + ".clstr") as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
                continue
            m = re.search(r">(\S+?)\.\.\.", line)
            if m:
                key2cl[m.group(1)] = cid
    return key2cl


def main(db_json, summary, outdir, cache16s):
    outdir, cache16s = Path(outdir), Path(cache16s)
    outdir.mkdir(parents=True, exist_ok=True)
    cache16s.mkdir(parents=True, exist_ok=True)
    data = json.load(open(db_json))
    ftp = ftp_paths(Path(summary))

    hits = []
    for recT, rec in data.items():
        ctx = rec.get("ctx_ssap", {})
        ctx0 = ctx.get("ctx_0") or []
        if not ctx0:
            continue
        seq = ctx0[0].get("sequence", "")
        if not (LEN_MIN <= len(seq) <= LEN_MAX):
            continue
        pfams = [e for sub in (ctx.get("context_elements") or []) for e in sub]
        has_exo = any(e in EXO for e in pfams)
        for key, ori in rec.items():
            if not str(key).startswith("ori_"):
                continue
            for sk, ssb in ori.items():
                if not str(sk).startswith("ssb_"):
                    continue
                s0 = ssb.get("ctx_0") or []
                if not s0:
                    continue
                if not s0[0].get("sequence", "").endswith(MOTIF):
                    continue
                if ssb.get("status") != "host":
                    continue
                hits.append({
                    "recT": ctx0[0].get("accession") or recT,
                    "host": ori.get("strain", ""),
                    "taxid": str(ori.get("TaxID", "")),
                    "lineage": ";".join(ori.get("linage") or []),
                    "assembly": ori.get("Accession", ""),
                    "length": len(seq),
                    "has_recE_partner": has_exo,
                })

    exo = [h for h in hits if h["has_recE_partner"]]
    # 16S rDNA per host assembly of the RecE-paired candidates
    seqs = {}
    for h in exo:
        gcf = h["assembly"]
        if not gcf or gcf in seqs:
            continue
        if gcf not in ftp:
            print(f"[funnel] {gcf} not in assembly_summary_refseq.txt", file=sys.stderr)
            continue
        s = download_16s(gcf, ftp[gcf][0], ftp[gcf][1], cache16s)
        if s:
            seqs[gcf] = s
    print(f"[funnel] 16S sequences retrieved for {len(seqs)} assemblies")

    res = {"motif": MOTIF, "recTdb_records": len(data),
           "step_ssb_c7_hits": len(hits),
           "step_ssb_c7_unique_recT": len({h["recT"] for h in hits}),
           "step_ssb_c7_unique_hosts": len({h["taxid"] for h in hits}),
           "step_cognate_recE_pairs": len(exo),
           "step_cognate_unique_recT": len({h["recT"] for h in exo}),
           "step_cognate_unique_hosts": len({h["taxid"] for h in exo}),
           "step_cognate_unique_assemblies": len({h["assembly"] for h in exo}),
           "exo_pfam_list": EXO}
    for ident in (0.99, 0.97, 0.95):
        key2cl = cluster16s(seqs, outdir, ident)
        missing = {h["assembly"] for h in exo} - set(seqs)
        res[f"step_unique_hosts_16S_{int(ident*100)}"] = len(set(key2cl.values())) + len(missing)
    res["assemblies_without_16S"] = len({h["assembly"] for h in exo} - set(seqs))

    with open(outdir / "funnel_candidates.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hits[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sorted(hits, key=lambda h: (not h["has_recE_partner"], h["lineage"], h["host"])))

    json.dump(res, open(outdir / "funnel.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:5])

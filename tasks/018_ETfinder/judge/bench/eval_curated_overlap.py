"""Evaluate fixed ETfinder host candidates against the published 91 records.

Candidate definition: Results, 'Mining RecET homolog candidates from
alpha-proteobacteria for R. sphaeroides', Fig. 2A, and Supplementary Dataset S1.
Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN) are the
user-specified benchmark extension, NOT equations reported in the paper.
Taxonomy queries transcribe ETE3 NCBITaxa's scientific-name/synonym/lineage
lookups using its existing database in SQLite read-only mode.
"""
import collections
import csv
import functools
import json
import re
import sqlite3
import sys
from pathlib import Path

from reuse_outputs import sha256

REFERENCE = 'paper/supp/S1/ETfinder_Rspcase/rsp_results/unique16s_rect_fix.txt'


def normalize(name):
    """Strip strain suffixes of named species, retain identifiers after 'sp.'."""
    words = re.sub(r'[\[\]"\u201c\u201d]', '', name.lower()).split()
    if len(words) < 2:
        return ' '.join(words)
    offset = 1 if words[0] == 'candidatus' else 0
    if len(words) > offset + 1 and words[offset + 1] not in ('sp.', 'sp', 'bacterium'):
        words = words[:offset + 2]
    return ' '.join(words)


class Taxonomy:
    def __init__(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')

    @functools.lru_cache(None)
    def taxon(self, taxid):
        row = self.db.execute('SELECT taxid,spname,rank,track FROM species WHERE taxid=?',
                              (int(taxid),)).fetchone()
        if not row:
            merged = self.db.execute('SELECT taxid_new FROM merged WHERE taxid_old=?',
                                     (int(taxid),)).fetchone()
            if merged:
                return self.taxon(merged[0])
        return row

    @functools.lru_cache(None)
    def names(self, name):
        ids = [r[0] for r in self.db.execute('SELECT taxid FROM species WHERE spname=?', (name,))]
        if not ids:
            ids = [r[0] for r in self.db.execute('SELECT taxid FROM synonym WHERE spname=?', (name,))]
        return tuple(sorted(set(ids)))

    @functools.lru_cache(None)
    def ranked(self, taxid, rank):
        row = self.taxon(taxid)
        if row:
            for ancestor in map(int, row[3].split(',')):  # ETE stores leaf first
                item = self.taxon(ancestor)
                if item and item[2] == rank:
                    return {'taxid': item[0], 'name': item[1]}
        return None

    def annotate(self, record):
        ids = (int(record['taxid']),) if record.get('taxid') else self.names(record['host'])
        source = 'input_taxid' if record.get('taxid') else 'exact_name_or_synonym'
        if not ids:
            ids = self.names(normalize(record['host']))
            source = 'normalized_name'
        record['resolved_taxids'] = list(ids)
        record['resolution_source'] = source if ids else 'unresolved'
        record['normalized_name'] = normalize(record['host'])
        for rank in ('species', 'genus'):
            taxa = {v['taxid']: v for t in ids if (v := self.ranked(t, rank))}
            record[rank] = next(iter(taxa.values())) if len(taxa) == 1 else None
        # Only trust a textual genus if the name is a binomial or Genus sp.
        words = record['normalized_name'].split()
        genus = None
        if len(words) >= 2 and words[1] != 'bacterium' and words[0] not in ('uncultured', 'unclassified'):
            genus = words[1] if words[0] == 'candidatus' else words[0]
        record['normalized_genus'] = genus
        return record


def reconstruct_candidates(folder):
    with (folder / 'funnel_candidates.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    paired = [r for r in rows if r['has_recE_partner'].lower() == 'true']
    by_assembly = collections.defaultdict(list)
    for row in paired:
        by_assembly[row['assembly']].append(row)
    clusters, current = [], None
    for line in (folder / 'host16S_99_cdhit.clstr').read_text().splitlines():
        if line.startswith('>Cluster'):
            current = {'id': line[1:], 'assemblies': [], 'representative': None}
            clusters.append(current)
        else:
            m = re.search(r'>(\S+)\.\.\.', line)
            if m:
                assert current is not None
                current['assemblies'].append(m[1])
                if line.rstrip().endswith('*'):
                    assert current['representative'] is None
                    current['representative'] = m[1]
    clustered = {a for c in clusters for a in c['assemblies']}
    assert clustered <= by_assembly.keys()
    missing = sorted(by_assembly.keys() - clustered)
    units = [(c['id'], c['representative'], c['assemblies']) for c in clusters]
    units += [('missing_16S:' + a, a, [a]) for a in missing]
    result = []
    for identifier, assembly, members in units:
        assert assembly in by_assembly
        metadata = by_assembly[assembly]
        assert len({r['taxid'] for r in metadata}) == 1
        row = dict(sorted(metadata, key=lambda r: (r['recT'], r['host']))[0])
        row.update(id=identifier, cluster_members=members,
                   recT_accessions=sorted({r['recT'] for r in metadata}))
        result.append(row)
    funnel = json.loads((folder / 'funnel.json').read_text())
    assert len(result) == funnel['step_unique_hosts_16S_99']
    assert len(missing) == funnel['assemblies_without_16S']
    audit = {'tsv_rows': len(rows), 'paired_rows': len(paired), 'paired_assemblies': len(by_assembly),
             'saved_99_percent_clusters': len(clusters), 'assemblies_without_16S': len(missing),
             'selection': 'Saved CD-HIT * representative per 99% cluster, plus each missing-16S assembly; independent of reference.',
             'precluster_exact_unique_name_overlap': None}
    return result, audit, paired


def match_reason(p, r, rank, mode):
    a, b = p[rank], r[rank]
    if mode != 'name_only' and a and b:
        return 'taxid' if a['taxid'] == b['taxid'] else None
    if mode == 'taxid_only':
        return None
    key = 'normalized_name' if rank == 'species' else 'normalized_genus'
    if p[key] and p[key] == r[key]:
        return 'normalized_name'
    return None


def compare(pred, ref, rank, mode='taxid_then_name'):
    # Maximum bipartite matching: no prediction can cover multiple references.
    # With rank-resolved IDs, edges are disjoint complete identity groups.
    edges = [[j for j, r in enumerate(ref) if match_reason(p, r, rank, mode)] for p in pred]
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in range(len(pred)):
        augment(i, set())
    used = set(owners.values())
    tp = len(owners)
    fp, fn = len(pred) - tp, len(ref) - tp
    return {'n_predicted': len(pred), 'n_reference': len(ref), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / len(pred) if pred else 0.0,
            'recall': tp / len(ref) if ref else 0.0,
            'f1': 2 * tp / (len(pred) + len(ref)) if pred or ref else 0.0,
            'std_error': 0.0,
            'matched': [{'predicted': pred[i], 'reference': ref[j],
                         'match_by': match_reason(pred[i], ref[j], rank, mode)}
                        for j, i in sorted(owners.items())],
            'unmatched_reference': [r for j, r in enumerate(ref) if j not in owners],
            'unmatched_predicted': [p for i, p in enumerate(pred) if i not in used]}


def deduplicate(records, rank):
    unique = {}
    for row in records:
        key = ('taxid', row[rank]['taxid']) if row[rank] else ('unresolved', row['id'])
        unique.setdefault(key, row)
    return list(unique.values())


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main(root, run):
    root = Path(root)
    out = root / 'results/metric' / run
    assert not (out / 'metric.json').exists(), 'Use a new run name for a clean evaluation.'
    manifest = json.loads((out / 'reuse_manifest.json').read_text())
    for entry in manifest['files']:
        assert sha256(out / entry['snapshot']) == entry['sha256'], entry['snapshot']
    pred, audit, paired = reconstruct_candidates(out / 'inputs/results/funnel')
    print(f'Verified reused outputs; reconstructed {len(pred)} candidates.', flush=True)
    reference_path = root / REFERENCE
    ref = []
    for line in reference_path.read_text().splitlines():
        if line.startswith('>'):
            accession, *name = line[1:].split(maxsplit=1)
            ref.append({'id': accession, 'accession': accession, 'host': name[0] if name else ''})
    assert len(ref) == 91 and len({r['id'] for r in ref}) == 91
    audit['reference_unique_names'] = len({r['host'] for r in ref})
    audit['reference_records_with_names'] = sum(bool(r['host']) for r in ref)
    audit['precluster_exact_unique_name_overlap'] = len({r['host'] for r in ref} & {p['host'] for p in paired})
    db_path = root / '.home/.etetoolkit/taxa.sqlite'
    taxonomy = Taxonomy(db_path)
    pred = [taxonomy.annotate(p) for p in pred]
    ref = [taxonomy.annotate(r) for r in ref]
    unresolved = [r for r in ref if not r['species']]
    print(f'Resolved {len(ref)-len(unresolved)}/{len(ref)} reference species; computing overlaps and database fingerprint.', flush=True)
    mode = 'name_only' if len(unresolved) > 20 else 'taxid_then_name'
    scores = {rank: compare(pred, ref, rank, mode) for rank in ('species', 'genus')}
    size_ok = 45 <= len(pred) <= 182
    result = {
        'task_metric': 'species-level host-record F1 against the published 91-host curated list',
        'definition': 'Maximum one-to-one host-record matching by species taxid, falling back to normalized names only when rank taxid is unavailable; P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN).',
        'metric_source': 'Benchmark-defined overlap formulas; paper does not report precision/recall/F1.',
        'paper_candidate_source': 'Results: Mining RecET homolog candidates from alpha-proteobacteria for R. sphaeroides; Fig. 2A; Supplementary Dataset S1.',
        'paper_value': None, 'paper_candidate_count': 91, 'reference_self_f1': 1.0,
        'n_predicted': len(pred), 'n_reference': len(ref),
        'counting_unit': 'One saved 16S cluster representative (or missing-16S assembly) per prediction, one FASTA accession per reference. Repeated taxids are retained and matched one-to-one to preserve the specified 88/91 denominators.',
        'primary_matching': mode, **scores,
        'candidate_set_size_constraint': {'minimum': 45, 'maximum': 182, 'value': len(pred), 'satisfied': size_ok},
        'candidate_set_size_constraint_satisfied': size_ok,
        'constrained_score': scores['species']['f1'] if size_ok else 0.0,
        'matched': scores['species']['matched'],
        'unmatched_reference': scores['species']['unmatched_reference'],
        'unmatched_predicted': scores['species']['unmatched_predicted'],
        'taxid_only': {rank: compare(pred, ref, rank, 'taxid_only') for rank in ('species', 'genus')},
        'name_only': {rank: compare(pred, ref, rank, 'name_only') for rank in ('species', 'genus')},
        'deduplicated_taxon_sets': {rank: compare(deduplicate(pred, rank), deduplicate(ref, rank), rank)
                                    for rank in ('species', 'genus')},
        'taxonomy_resolution': {'reference_species_resolved': len(ref) - len(unresolved),
                                'reference_species_unresolved': unresolved,
                                'predicted_species_resolved': sum(p['species'] is not None for p in pred),
                                'reference_genus_resolved': sum(r['genus'] is not None for r in ref),
                                'predicted_genus_resolved': sum(p['genus'] is not None for p in pred)},
        'std_error': 0.0,
        'uncertainty': 'Deterministic census of fixed saved outputs, no random seeds or Monte Carlo; computational standard error is zero for all counts, ratios, and score. This is not an estimate of biological or taxonomic uncertainty.',
        'reductions': ['name-only primary: more than 20 unresolved reference species'] if mode == 'name_only' else [],
        'method_unchanged': True, 'input_audit': audit,
        'provenance': {'run': run, 'reference': REFERENCE, 'reference_sha256': sha256(reference_path),
                       'taxonomy_db': str(db_path.relative_to(root)), 'taxonomy_sha256': sha256(db_path),
                       'taxonomy_size_bytes': db_path.stat().st_size, 'reuse_manifest': 'reuse_manifest.json'},
        'normalization': 'Lowercase and remove brackets/extra whitespace; named binomials lose strain suffixes. Preserve designators after sp. or bacterium to avoid collapsing distinct unnamed species. Taxonomic synonyms and species/genus ancestors come from the frozen ETE3 DB. Resolved taxid disagreements never fall back to name matches.',
        'limitations': ['Species-ranked generic NCBI taxa such as Mesorhizobium sp. are matched literally; headers do not identify their underlying biological species.',
                       'Genus-unknown bacterium labels are not treated as genus names; unresolved entries remain in denominators.',
                       'The TSV contains 637 pre-cluster records, not 88 final representatives. Saved 99% cluster membership is required to recover the final candidate identities.']}
    atomic_json(out / 'metric.json', result)
    atomic_json(out / 'metrics_v2.json', result)
    print('Level          TP  FP  FN  Precision   Recall       F1')
    for rank, s in scores.items():
        print(f'{rank:13} {s["tp"]:3} {s["fp"]:3} {s["fn"]:3} {s["precision"]:10.6f} {s["recall"]:8.6f} {s["f1"]:8.6f}')
    print(f'Candidates: {len(pred)}; paper: 91; constraint 45..182: {size_ok}')
    print(f'Constrained score: {result["constrained_score"]:.8f}; paper F1: not reported; reference self-F1: 1.0')
    print(f'Reference species resolved: {len(ref)-len(unresolved)}/91; primary: {mode}; SE=0 (deterministic)')
    for rank, s in result['deduplicated_taxon_sets'].items():
        print(f'Deduplicated {rank} set: pred={s["n_predicted"]}, ref={s["n_reference"]}, TP={s["tp"]}, F1={s["f1"]:.6f}')


if __name__ == '__main__':
    main(*sys.argv[1:])

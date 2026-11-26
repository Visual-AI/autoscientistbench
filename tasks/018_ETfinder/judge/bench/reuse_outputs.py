"""Snapshot the explicitly authorized existing outputs; no ETfinder rerun."""
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(root, run):
    root = Path(root)
    out = root / 'results/metric' / run
    out.mkdir(parents=True, exist_ok=False)
    sources = [root / 'results/funnel' / f for f in (
        'funnel_candidates.tsv', 'funnel.json', 'host16S_99_cdhit.clstr')]
    sources += sorted(p for p in (root / 'results/lddeipf_rep/output').rglob('*') if p.is_file())
    assert sources and all(p.is_file() for p in sources)
    entries = []
    for src in sources:
        relative = src.relative_to(root)
        dst = out / 'inputs' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        shutil.copyfile(src, dst)
        assert sha256(dst) == before
        entries.append({'source': str(relative), 'snapshot': str(dst.relative_to(out)),
                        'sha256': before, 'bytes': src.stat().st_size})
    manifest = {'stage': 'reuse', 'method_rerun': False, 'run': run, 'files': entries}
    (out / 'reuse_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Reused {len(entries)} files ({sum(x["bytes"] for x in entries)} bytes); no training/simulation required.')


if __name__ == '__main__':
    main(*sys.argv[1:])

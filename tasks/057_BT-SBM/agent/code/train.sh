#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 2 && $1 == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
out="$2"
mkdir -p "$out/Rlib"
out="$(cd "$out" && pwd)"
build="$(mktemp -d "${TMPDIR:-/tmp}/sta32-build.XXXXXX")"
trap 'rm -rf "$build"' EXIT
cp -R BTSBM "$build/BTSBM"
# Recompile candidate C++ every run; never load the reproduction's installed BTSBM.
find "$build/BTSBM/src" -type f \( -name '*.o' -o -name '*.so' \) -delete
export MAKEFLAGS="-j${BENCH_CPUS:-1}"
R CMD INSTALL --preclean --no-docs --no-html --no-demo --library="$out/Rlib" "$build/BTSBM"
python3 - "$out" <<'PYCODE'
import hashlib,json,pathlib,sys
p=pathlib.Path('BTSBM'); out=pathlib.Path(sys.argv[1])
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file()}
(out/'build_manifest.json').write_text(json.dumps({'candidate_sources':manifest,'bench_seed':__import__('os').environ['BENCH_SEED']},indent=2)+'\n')
PYCODE

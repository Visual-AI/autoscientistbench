"""Protocol adapter; invoke the editable caller on the supplied observed alignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
inp, output = Path(a.input), Path(a.output)
manifest = json.loads((inp/'manifest.json').read_text())
cpus = int(os.environ['BENCH_CPUS'])
if not 1 <= cpus <= 32:
    raise ValueError('SNooPy benchmark permits at most 32 CPUs')
seed = int(os.environ['BENCH_SEED'])
code = Path(__file__).resolve().parent
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='snoopy-', dir=os.environ.get('TMPDIR')) as temp:
    out = Path(temp)/'calls'
    start = time.monotonic()
    result = subprocess.run([sys.executable, str(code/'invoke.py'), str(code/'snoopy.py'),
        '-r', str(inp/'assembly.fasta'), '-b', str(inp/'mapped.bam'), '-o', str(out),
        '-t', str(cpus), '--benchmark'], timeout=None)
    elapsed = time.monotonic()-start
    result.check_returncode()
    vcf = (out/'variants.vcf').read_text()
    prediction = {'version':1, 'contigs':manifest['contigs'], 'vcf':vcf,
        'runtime':{'wall_s':elapsed,'cpus':cpus,'exit_code':result.returncode,'seed':seed,
        'caller_sha256':hashlib.sha256((code/'snoopy.py').read_bytes()).hexdigest()}}
    output.write_text(json.dumps(prediction, allow_nan=False)+'\n')

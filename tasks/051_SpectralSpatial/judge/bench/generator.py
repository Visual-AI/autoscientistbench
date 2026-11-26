#!/usr/bin/env python3
"""Frozen generator wrapper. Arguments: OUTPUT_ROOT GRID_SIDE REPS SEED_OFFSET."""
from pathlib import Path
import os
import subprocess
import sys
root = Path(__file__).resolve().parent
runtime = Path(os.environ['BENCH_RUNTIME_DIR']) / 'r' / 'bin' / 'Rscript'
subprocess.run([str(runtime), str(root/'generator.R'), str(root), *sys.argv[1:]], check=True)

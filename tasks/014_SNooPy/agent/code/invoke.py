"""Seed and run the current candidate source, never an operator-frozen caller."""
import os
import random
import runpy
import sys
import numpy as np

seed = int(os.environ['BENCH_SEED'])
random.seed(seed)
np.random.seed(seed % (2**32))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')

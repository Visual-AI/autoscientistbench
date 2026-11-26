"""Disposable worker for GAM smoothing-parameter search."""
import pickle
import sys

from utils import _run_gam_gridsearch

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as stream:
        inputs = pickle.load(stream)
    print("READY", flush=True)
    result = _run_gam_gridsearch(*inputs)
    with open(sys.argv[2], "wb") as stream:
        pickle.dump(result, stream)

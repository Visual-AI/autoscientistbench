#!/usr/bin/env python3
import sys
from format import read, predictions
try:
    observed=read(sys.argv[2])
    ids=[row['id'] for row in observed['datasets']]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate input IDs")
    predictions(read(sys.argv[1]),ids)
    print("valid posterior probabilities")
except Exception as exc:
    print(str(exc),file=sys.stderr)
    sys.exit(1)

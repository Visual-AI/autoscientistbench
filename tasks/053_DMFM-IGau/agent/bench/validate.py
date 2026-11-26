#!/usr/bin/env python3
import sys
from protocol import load,rows
try:
    n=sum(1 for _ in rows(sys.argv[1],load(sys.argv[2])))
    print("ok: %d post-burn draws"%n)
except Exception as e:
    print("invalid: "+str(e),file=sys.stderr);sys.exit(1)

#!/usr/bin/env python3
import json,sys
from common import evaluate,load
if __name__=='__main__':
    print(json.dumps(evaluate(load(sys.argv[1]),load(sys.argv[2])),allow_nan=False))

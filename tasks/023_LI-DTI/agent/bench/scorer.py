#!/usr/bin/env python3
import json
import sys
from metric import read_rows, evaluate
try:
    result=evaluate(read_rows(sys.argv[1],'pred'),read_rows(sys.argv[2],'y'))
except Exception as exc:
    result={'AUPR':0.0,'valid':False,'error':str(exc)}
print(json.dumps(result,allow_nan=False))

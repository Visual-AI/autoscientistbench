#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_json,check_prediction
if __name__=="__main__":
 try:
  instances=read_json(Path(sys.argv[2])/"manifest.json")["instances"]
  check_prediction(read_json(sys.argv[1]),instances)
 except Exception as exc:sys.exit(str(exc))
 print("valid hFDR prediction")

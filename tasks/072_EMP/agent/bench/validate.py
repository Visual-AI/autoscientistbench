#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from certificate import load,validate_arrays

def main():
 try:
  p=load(sys.argv[1]); x=load(sys.argv[2]); validate_arrays(p,x['id'])
  print(f"ok: {len(p['id'])} rows and complete FC12 certificate")
 except Exception as e:
  print(f'invalid: {e}',file=sys.stderr); return 1
 return 0
if __name__=='__main__': sys.exit(main())

#!/usr/bin/env python3
"""validate.py PRED INPUT: exact schema, row identity, finiteness and geometry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mesh_io import read, matching, guards, PRED_COLUMNS, INPUT_COLUMNS

def main():
    p=read(sys.argv[1],PRED_COLUMNS); x=read(sys.argv[2],INPUT_COLUMNS)
    matching(p,x);guards(p)
    print(f'ok: {len(p)} admissible meshes')
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,OverflowError) as e:
        print(str(e),file=sys.stderr);sys.exit(1)

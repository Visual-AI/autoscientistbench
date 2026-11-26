#!/usr/bin/env python3
import json,sys,zipfile
from common import check,load,loads,sha
if __name__=='__main__':
    try:
        with zipfile.ZipFile(sys.argv[2]) as z:manifest=loads(z.read('manifest.json'))
        p=check(load(sys.argv[1]),manifest,sha(sys.argv[2]))
        print('ok:',len(p['query_ids']),'queries;',len(p['hsps']),'HSPs')
    except Exception as e:sys.exit('invalid prediction: '+str(e))

"""Trusted replay entry point; candidate code runs only in isolated workers."""
import json,sys
from private_score import evaluate
if __name__=='__main__':
 try: result=evaluate(*sys.argv[1:])
 except Exception as e: result={'ccv2':float('inf'),'trusted_oracle':False,'reason':str(e)}
 print(json.dumps(result))

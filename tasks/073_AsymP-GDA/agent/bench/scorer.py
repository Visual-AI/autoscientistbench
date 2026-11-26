"""Private replay entry point with native-library stdout kept out of JSON."""
import json,os,sys
if __name__=='__main__':
 sys.stdout.flush();saved=os.dup(1)
 try:
  with open(os.devnull,'w') as sink:
   os.dup2(sink.fileno(),1)
   try:
    from private_score import evaluate
    result=evaluate(*sys.argv[1:])
   except Exception as exc:result={'nashconv':float('inf'),'trusted_oracle':False,'reason':str(exc)}
 finally:os.dup2(saved,1);os.close(saved)
 print(json.dumps(result))

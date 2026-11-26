"""R line bridge to the task's bounded private RPC; contains no sampler data."""
import os,socket,sys
from pathlib import Path
from private_rpc import call
server=socket.socket(socket.AF_INET,socket.SOCK_STREAM);server.bind(('127.0.0.1',0));server.listen(1)
path=Path(sys.argv[1]);partial=path.with_suffix('.tmp');partial.write_text(str(server.getsockname()[1]));os.replace(partial,path)
connection,_=server.accept()
with connection,connection.makefile('r',encoding='ascii') as reader,connection.makefile('w',encoding='ascii') as writer:
 for line in reader:
  try:
   method,indices,end=line.rstrip('\n').split('\t')
   if method!='sample':raise ValueError('unknown method')
   out=call('sample',[int(x) for x in indices.split(',')],int(end))
   writer.write('ok\t'+','.join(map(str,out['old']))+'\t'+','.join(format(x,'.17g') for x in out['values'])+'\n');writer.flush()
  except Exception:
   writer.write('error\n');writer.flush();raise

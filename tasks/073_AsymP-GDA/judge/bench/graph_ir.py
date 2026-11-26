"""Public LiteEFG graph recorder. Emits data instructions, never executable code.

Use `import graph_ir as LiteEFG` when defining a candidate graph. Numeric
expressions retain LiteEFG semantics when interpreted by the trusted service.
"""
import contextlib
import os
_SEED=int(os.environ.get("BENCH_SEED", "1"))
_REMOTE_SEED=None
RANDOM_FUNCTIONS=("uniform","normal","exponential")
_CURRENT=None
BUILTINS=('action_set_size','subtree_size','utility','reach_prob','opponent_reach_prob')
FUNCTIONS=('const','aggregate','argmax','argmin','cat','copy','dot','euclidean','exp','log','max','maximum','mean','min','minimum','negative_entropy','normalize','project','sum')
METHODS=('argmax','argmin','copy','dot','euclidean','exp','log','max','mean','min','negative_entropy','normalize','project','sum','inplace')
class Graph:
 def __init__(self):
  global _CURRENT
  self._instructions=[];self._context=None;self._builtins={};self._seed=_SEED;_CURRENT=self
 def __getattr__(self,name):
  if name not in BUILTINS:raise AttributeError(name)
  if name not in self._builtins:
   node=GraphNode(self,len(self._instructions));self._instructions.append({'kind':'builtin','name':name});self._builtins[name]=node
  return self._builtins[name]
 def _emit(self,kind,name,args,kwargs):
  if self._context is None:raise ValueError('graph operations require backward/forward context')
  node=GraphNode(self,len(self._instructions))
  self._instructions.append({'kind':kind,'name':name,'args':_encode(args,self),'kwargs':_encode(kwargs,self),'context':dict(self._context)})
  return node
 def to_ir(self,policy):
  policies=policy if isinstance(policy,list) else [policy]
  if len(policies) not in (1,2) or any(not isinstance(p,GraphNode) or p.graph is not self for p in policies):raise ValueError('policy belongs to another graph')
  indices=[p.index for p in policies]
  return {'version':1,'instructions':self._instructions,'policy':indices if isinstance(policy,list) else indices[0]}
class GraphNode:
 def __init__(self,graph,index):self.graph,self.index=graph,index
 def _op(self,name,*args):return self.graph._emit('operator',name,(self,*args),{})
 def __add__(self,x):return self._op('add',x)
 def __radd__(self,x):return self._op('radd',x)
 def __sub__(self,x):return self._op('sub',x)
 def __rsub__(self,x):return self._op('rsub',x)
 def __mul__(self,x):return self._op('mul',x)
 def __rmul__(self,x):return self._op('rmul',x)
 def __truediv__(self,x):return self._op('truediv',x)
 def __rtruediv__(self,x):return self._op('rtruediv',x)
 def __pow__(self,x):return self._op('pow',x)
 def __rpow__(self,x):return self._op('rpow',x)
 def __neg__(self):return self._op('neg')
 def __eq__(self,x):return self._op('eq',x)
 def __lt__(self,x):return self._op('lt',x)
 def __le__(self,x):return self._op('le',x)
 def __gt__(self,x):return self._op('gt',x)
 def __ge__(self,x):return self._op('ge',x)
 def __bool__(self):raise TypeError('GraphNodes are symbolic, not Python booleans')
 def __getattr__(self,name):
  if name not in METHODS:raise AttributeError(name)
  def method(*args,**kwargs):
   node=self.graph._emit('method',name,(self,*args),kwargs)
   return None if name=='inplace' else node
  return method
# These are annotation aliases only. They cannot open a native game.
class Environment:pass

def _encode(value,graph):
 if isinstance(value,GraphNode):
  if value.graph is not graph:raise ValueError('cross-graph node reference')
  return {'node':value.index}
 if isinstance(value,(list,tuple)):return [_encode(x,graph) for x in value]
 if isinstance(value,dict):return {k:_encode(v,graph) for k,v in value.items()}
 return value
@contextlib.contextmanager
def _context(direction,is_static=False,color=0):
 if _CURRENT is None or _CURRENT._context is not None:raise ValueError('graph context missing or nested')
 graph=_CURRENT;graph._context={'direction':direction,'is_static':is_static,'color':color}
 try:yield
 finally:graph._context=None
def backward(is_static=False,color=0):return _context('backward',is_static,color)
def forward(is_static=False,color=0):return _context('forward',is_static,color)
def _function(name):
 def operation(*args,**kwargs):
  if _CURRENT is None:raise ValueError('construct a Graph first')
  return _CURRENT._emit('function',name,args,kwargs)
 return operation
for _name in FUNCTIONS:globals()[_name]=_function(_name)

class _Random:
 def __getattr__(self,name):
  if name not in RANDOM_FUNCTIONS:raise AttributeError(name)
  def operation(*args,**kwargs):
   if _CURRENT is None:raise ValueError('construct a Graph first')
   return _CURRENT._emit('random',name,args,kwargs)
  return operation
random=_Random()

def set_seed(seed):
 """Mirror the pinned native RNG seed, including after opening a run."""
 global _SEED
 if type(seed) is not int or not -(2**31)<=seed<2**31:raise ValueError('seed must fit native int32')
 _SEED=seed
 if _REMOTE_SEED is not None:_REMOTE_SEED(seed)

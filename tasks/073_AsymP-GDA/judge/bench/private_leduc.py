"""Frozen counted LiteEFG interface. Private payload and FileEnv stay in closures.

One Figure-4 update is a full two-player alternating sweep = two primitive
LiteEFG update traversals. Counters include all independently created runs.
The process boundary is not a security boundary: bypasses require patch audit.
"""
import os
import tempfile
import numpy as np
import LiteEFG

MAX_UPDATES=1_000_000


def _load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def instances(path):
    """Only opaque instance identifiers; no game parameters or seeds."""
    with np.load(path,allow_pickle=False) as z:return [str(x) for x in z['id']]


def _file_env(text):
    fd,path=tempfile.mkstemp(suffix='.openspiel')
    try:
        with os.fdopen(fd,'w') as f:f.write(text)
        return LiteEFG.FileEnv(path,traverse_type='Enumerate')
    finally:os.unlink(path)


def _event(kind, **fields):
    """No-op instrumentation seam; the independent replay probe replaces it."""
    pass


def open_run(path, iid, graph, run_id):
    """Bind an algorithm graph; return only counted update and strategy export.

    Allowed graph reads are LiteEFG graph operations inside update traversals.
    Export reads the current strategy, without an extra payoff traversal.
    """
    data=_load(path);ids=[str(x) for x in data['id']]
    env=_file_env(str(data['game'][ids.index(iid)]));env.set_graph(graph)
    traversals=0
    _event('open',iid=iid,run_id=run_id)
    class Oracle:
        __slots__=()
        def update(self,*args,**kwargs):
            nonlocal traversals
            # Arbitrary graph colors are allowed, but one explicit player per traversal.
            player=kwargs.get('upd_player')
            if player not in (1,2):raise ValueError('update requires upd_player=1 or 2')
            traversals+=1
            _event('update',iid=iid,run_id=run_id,player=player)
            if traversals>2*MAX_UPDATES:raise RuntimeError('traversal budget exceeded')
            return env.update(*args,**kwargs)
        def export(self,node):
            players=[env.get_value(p,node) for p in (1,2)]
            _event('export',iid=iid,run_id=run_id,traversals=traversals,players=players)
            return dict(players=players,traversals=traversals,updates=traversals/2,oracle=1)
    return Oracle()


class PolicyGraph(LiteEFG.Graph):
    def __init__(self):
        super().__init__()
        with LiteEFG.backward(is_static=True):
            self.u=LiteEFG.const(self.action_set_size,1.0/self.action_set_size)


def schema_for(text):
    graph=PolicyGraph();env=_file_env(text);env.set_graph(graph)
    return [[str(p),str(j),name,str(a)] for p in (1,2)
            for j,(name,values) in enumerate(env.get_value(p,graph.u)) for a in range(len(values))]


def evaluate(text, policies):
    """Trusted scoring only: no candidate code imported; final pair's NashConv."""
    graph=PolicyGraph();env=_file_env(text);env.set_graph(graph)
    for p in (1,2):
        shape=env.get_value(p,graph.u)
        values=[[policies[(p,j,a)] for a in range(len(v))] for j,(_,v) in enumerate(shape)]
        env.set_value(p,graph.u,values)
    return sum(env.exploitability(graph.u))

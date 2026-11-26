"""Counted RPC client; native graph state and games remain in the scorer."""
import json
from pathlib import Path
import graph_ir
from graph_ir import GraphNode
from private_rpc import call
MAX_UPDATES = 1_000_000
_SESSION_STARTED = False


def instances(path):
    data=json.loads(Path(path).read_text())
    return [str(i) for i in data.get('ids',[data.get('id')])]


class Oracle:
    def __init__(self,graph,policy,handle):
        self.graph,self.policy,self.handle = graph,policy,handle

    def _node(self,node):
        if isinstance(node,(list,tuple)):
            return [self._node(x) for x in node]
        if not isinstance(node,GraphNode) or node.graph is not self.graph:
            raise ValueError('foreign graph node')
        return node.index

    def update(self,node,*,upd_player,upd_color=None,traverse_type='default'):
        call('update',self.handle,self._node(node),upd_player,
                    [-1] if upd_color is None else upd_color,traverse_type)

    def update_many(self,updates,repeats=1):
        commands=[[self._node(node),player,colors] for node,player,colors in updates]
        return call('update_many',self.handle,commands,repeats)

    def snapshot(self,node):
        return call('snapshot',self.handle,self._node(node))

    def export(self,node):
        """Read current strategies; later updates require a new final export."""
        return call('export',self.handle,self._node(node))


def open_run(path,iid,graph,run_id,policy=None):
    global _SESSION_STARTED
    if not _SESSION_STARTED:
        call('seed',graph_ir._SEED)
        graph_ir._REMOTE_SEED=lambda seed:call('seed',seed)
        _SESSION_STARTED=True
    policy=graph.u if policy is None else policy
    ir=graph.to_ir(policy)
    # Streaming prevents the transport's per-frame bound imposing a graph-size
    # bound. Small graphs keep the existing single-call path.
    if len(ir['instructions']) <= 2048:
        result=call('open',str(iid),int(run_id),ir)
    else:
        upload=call('graph_begin',str(iid),int(run_id),ir['policy'])
        for start in range(0,len(ir['instructions']),256):
            call('graph_append',upload,ir['instructions'][start:start+256])
        result=call('graph_commit',upload)
    return Oracle(graph,policy,result['handle'])

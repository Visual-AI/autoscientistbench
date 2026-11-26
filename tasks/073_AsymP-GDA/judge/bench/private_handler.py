"""Private game execution with one budget across every independently opened graph."""
import math
import threading
import multiprocessing
import time
import LiteEFG
from private_graph import compile_graph
from private_leduc import _file_env


class Budget:
    def __init__(self,max_updates,shared=False,deadline=None):
        self.limit = 2*max_updates
        self._shared = multiprocessing.get_context('fork').Value('q',0) if shared else None
        self._used = 0
        self.lock = self._shared.get_lock() if shared else threading.Lock()
        self.deadline = deadline

    @property
    def used(self):
        return self._shared.value if self._shared is not None else self._used

    def check_time(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError('overall evaluation budget exhausted')

    def consume(self,count):
        with self.lock:
            if type(count) is not int or count < 0 or self.used+count > self.limit:
                raise ValueError('combined traversal budget exceeded')
            self.check_time()
            if self._shared is not None:
                self._shared.value += count
            else:
                self._used += count


class Handler:
    def __init__(self,iid,run_id,text,budget,seed=1):
        self.iid,self.run_id,self.text,self.budget = iid,run_id,text,budget
        self.started = self.finished = self.invalid = False
        self.traversals = 0
        self.record = None
        self.runs = {}
        self.staging = {}
        self.next_handle = 0
        self.seed = seed
        self.seeded = False

    def _bind(self,iid,run_id,ir):
        if iid != self.iid or type(run_id) is not int:
            raise ValueError('graph belongs to a different instance')
        if not self.seeded:
            LiteEFG.set_seed(self.seed)
            self.seeded = True
        graph,nodes,policy = compile_graph(ir)
        env = _file_env(self.text)
        env.set_graph(graph)
        handle = self.next_handle
        self.next_handle += 1
        self.runs[handle] = dict(graph=graph,nodes=nodes,policy=policy,env=env,run_id=run_id)
        if not self.started:
            # Compatibility for existing operator checks of the original handle.
            self.graph,self.nodes,self.policy,self.env = graph,nodes,policy,env
        self.started = True
        self.finished = False
        self.record = None
        return {'id':iid,'run_id':run_id,'handle':handle,'max_traversals':self.budget.limit}

    def _command(self,run,node,player,colors,traverse_type='default'):
        indices = node if isinstance(node,list) else [node]
        if not indices or any(type(n) is not int or not 0 <= n < len(run['nodes']) or not isinstance(run['nodes'][n],LiteEFG.GraphNode) for n in indices):
            raise ValueError('invalid strategy node(s)')
        if type(player) is not int or player not in (1,2):
            raise ValueError('one explicit player per traversal required')
        if not isinstance(colors,list) or any(type(c) is not int or not -(2**31) <= c < 2**31 for c in colors):
            raise ValueError('invalid native graph colors')
        if traverse_type not in ('default','Enumerate'):
            raise ValueError('only the original Enumerate traversal is allowed')
        nodes = [run['nodes'][n] for n in indices]
        return (nodes if isinstance(node,list) else nodes[0],player,colors)

    def _read_policy(self,run,node):
        if node != run['policy']:
            raise ValueError('only the declared current policy can be exported')
        indices = node if isinstance(node,list) else [node,node]
        if len(indices) == 1:
            indices = indices*2
        players = [run['env'].get_value(p,run['nodes'][indices[p-1]]) for p in (1,2)]
        for player in players:
            for name,values in player:
                if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-10:
                    raise ValueError('only finite simplex policies may be exported')
        return {'players':players,'traversals':self.traversals,'updates':self.traversals/2,'oracle':1}

    def __call__(self,method,args,kwargs):
        try:
            self.budget.check_time()
            if kwargs:
                raise ValueError('keyword RPC arguments are unsupported')
            if method == 'seed' and len(args) == 1:
                seed = args[0]
                if type(seed) is not int or not -(2**31) <= seed < 2**31:
                    raise ValueError('seed must fit native int32')
                LiteEFG.set_seed(seed)
                self.seeded = True
                return True
            if method == 'start':
                if self.started or len(args) != 3 or args[:2] != [self.iid,self.run_id]:
                    raise ValueError('legacy start cannot reset an existing graph')
                return self._bind(*args)
            if method == 'open' and len(args) == 3:
                return self._bind(*args)
            # Large legal graphs can be uploaded in bounded RPC chunks; graph
            # size is governed by the original task memory/time budget.
            if method == 'graph_begin' and len(args) == 3:
                iid,run_id,policy = args
                if iid != self.iid or type(run_id) is not int:
                    raise ValueError('invalid graph upload identity')
                handle = self.next_handle
                self.next_handle += 1
                self.staging[handle] = [iid,run_id,dict(version=1,instructions=[],policy=policy)]
                return handle
            if method == 'graph_append' and len(args) == 2:
                handle,part = args
                if type(handle) is not int or handle not in self.staging or not isinstance(part,list):
                    raise ValueError('invalid graph upload')
                self.staging[handle][2]['instructions'].extend(part)
                return True
            if method == 'graph_commit' and len(args) == 1:
                handle = args[0]
                if type(handle) is not int or handle not in self.staging:
                    raise ValueError('unknown graph upload')
                return self._bind(*self.staging.pop(handle))
            if not self.started:
                raise ValueError('run is not active')
            # New clients address explicit handles; v2 default-handle calls stay
            # valid. Creating a distinct graph never resets the shared budget.
            lengths = {'update':4,'update_many':2,'snapshot':1,'export':1}
            if method not in lengths:
                raise ValueError('unknown oracle operation')
            if len(args) == lengths[method]+1:
                handle,args = args[0],args[1:]
            elif len(args) == lengths[method]:
                handle = min(self.runs)
            else:
                raise ValueError('malformed oracle operation')
            if type(handle) is not int or handle not in self.runs:
                raise ValueError('unknown graph handle')
            run = self.runs[handle]
            if method == 'update':
                command = self._command(run,*args)
                self.budget.consume(1)
                run['env'].update(command[0],upd_player=command[1],upd_color=command[2],traverse_type='Enumerate')
                self.traversals += 1
                self.record,self.finished = None,False
                return {'traversals':self.traversals}
            if method == 'update_many':
                commands,repeats = args
                if not isinstance(commands,list) or not commands or type(repeats) is not int or repeats < 1:
                    raise ValueError('invalid update batch')
                if any(not isinstance(x,list) or len(x) != 3 for x in commands):
                    raise ValueError('invalid update command')
                commands = [self._command(run,*x) for x in commands]
                count = len(commands)*repeats
                self.budget.consume(count)
                self.record,self.finished = None,False
                for _ in range(repeats):
                    for node,player,colors in commands:
                        self.budget.check_time()
                        run['env'].update(node,upd_player=player,upd_color=colors,traverse_type='Enumerate')
                        self.traversals += 1
                return {'traversals':self.traversals}
            record = self._read_policy(run,args[0])
            if method == 'export':
                self.record,self.finished = record,True
            return record
        except Exception:
            self.invalid = True
            raise

    def accepted_record(self):
        if self.invalid or not self.finished or self.staging:
            raise ValueError('unfinished or invalid private run')
        return self.record

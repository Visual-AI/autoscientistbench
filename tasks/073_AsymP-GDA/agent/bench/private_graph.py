"""Declarative native LiteEFG compiler; bounded by the task resource budget.

Only the pinned library's numeric graph vocabulary is callable. Candidate code,
Python attributes and game/environment access are never interpreted here.
"""
import math
import operator
import LiteEFG
from graph_ir import BUILTINS, FUNCTIONS, METHODS, RANDOM_FUNCTIONS

OPERATORS = {
    'add': operator.add, 'radd': lambda a,b:b+a,
    'sub': operator.sub, 'rsub': lambda a,b:b-a,
    'mul': operator.mul, 'rmul': lambda a,b:b*a,
    'truediv': operator.truediv, 'rtruediv': lambda a,b:b/a,
    'pow': operator.pow, 'rpow': lambda a,b:b**a,
    'neg': operator.neg, 'eq': operator.eq, 'lt': operator.lt,
    'le': operator.le, 'gt': operator.gt, 'ge': operator.ge,
}
KEYWORDS = {'size','val','aggregator','object','player','padding','distance',
            'gamma','mu','shifted','p_norm','ignore_negative','nodes',
            'lower','upper','mean','stddev','lambda_','arg0','arg1','node'}


def compile_graph(ir):
    required = {'version','instructions','policy'}
    if not isinstance(ir,dict) or not required <= set(ir) or set(ir)-required-{'seed'} or ir['version'] != 1:
        raise ValueError('invalid graph IR version/schema')
    instructions, policy = ir['instructions'], ir['policy']
    if not isinstance(instructions,list) or not instructions:
        raise ValueError('graph instructions must be nonempty')
    policies = policy if isinstance(policy,list) else [policy]
    if len(policies) not in (1,2) or any(type(p) is not int or not 0 <= p < len(instructions) for p in policies):
        raise ValueError('invalid graph policy')
    if 'seed' in ir:
        seed = ir['seed']
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError('seed must fit native int32')
        LiteEFG.set_seed(seed)
    graph, nodes, taints = LiteEFG.Graph(), [], []

    def decode(value):
        if isinstance(value,dict):
            if set(value) != {'node'} or type(value['node']) is not int or not 0 <= value['node'] < len(nodes):
                raise ValueError('invalid/backward-only graph reference')
            n = value['node']
            if nodes[n] is None:
                raise ValueError('inplace instruction is not a value')
            return nodes[n], taints[n]
        if isinstance(value,list):
            values = [decode(x) for x in value]
            return [x[0] for x in values], any(x[1] for x in values)
        if value is None or isinstance(value,bool):
            return value, False
        if type(value) in (int,float):
            if not math.isfinite(value):
                raise ValueError('nonfinite graph literal')
            return value, False
        if isinstance(value,str):
            return value, False
        raise ValueError('unsupported graph argument')

    for ins in instructions:
        if not isinstance(ins,dict):
            raise ValueError('invalid graph instruction')
        kind, name = ins.get('kind'), ins.get('name')
        if kind == 'builtin':
            if set(ins) != {'kind','name'} or name not in BUILTINS:
                raise ValueError('unknown graph builtin')
            nodes.append(getattr(graph,name))
            taints.append(name not in ('action_set_size','subtree_size'))
            continue
        if set(ins) != {'kind','name','args','kwargs','context'}:
            raise ValueError('invalid graph instruction schema')
        ctx = ins['context']
        if (not isinstance(ctx,dict) or set(ctx) != {'direction','is_static','color'}
                or ctx['direction'] not in ('backward','forward')
                or type(ctx['is_static']) is not bool
                or type(ctx['color']) is not int or not -(2**31) <= ctx['color'] < 2**31):
            raise ValueError('invalid native graph context')
        args, taint = decode(ins['args'])
        if not isinstance(args,list) or not isinstance(ins['kwargs'],dict):
            raise ValueError('invalid graph argument list')
        kwargs = {}
        for key,value in ins['kwargs'].items():
            if key not in KEYWORDS:
                raise ValueError('unsupported graph keyword')
            kwargs[key], t = decode(value)
            taint |= t
        if ctx['is_static'] and taint:
            raise ValueError('private utility/reach state is unavailable during free initialization')
        if kind == 'function':
            if name not in FUNCTIONS:
                raise ValueError('unknown graph function')
            # Scalar and symbolic sizes have native semantics. RLIMIT_AS and the
            # original solve deadline bound execution, not an invented 4096 cap.
            operation = getattr(LiteEFG,name)
        elif kind == 'random':
            if name not in RANDOM_FUNCTIONS:
                raise ValueError('unknown graph random operation')
            operation = getattr(LiteEFG.random,name)
        elif kind == 'method':
            if name not in METHODS or not args or not isinstance(args[0],LiteEFG.GraphNode):
                raise ValueError('unknown graph method')
            operation = getattr(args.pop(0),name)
        elif kind == 'operator':
            if name not in OPERATORS or kwargs or len(args) != (1 if name == 'neg' else 2):
                raise ValueError('unknown graph operator')
            operation = OPERATORS[name]
        else:
            raise ValueError('unknown instruction kind')
        with getattr(LiteEFG,ctx['direction'])(is_static=ctx['is_static'],color=ctx['color']):
            result = operation(*args,**kwargs)
        nodes.append(result)
        taints.append(taint)
    if any(not isinstance(nodes[p],LiteEFG.GraphNode) for p in policies):
        raise ValueError('policy is not a graph node')
    return graph, nodes, policy

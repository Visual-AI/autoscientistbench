"""Frozen TTN certificate parser and dense evaluator; no candidate imports."""
from collections import Counter, defaultdict
from pathlib import Path
import zipfile
import numpy as np


def need(value, message):
    if not value: raise ValueError(message)


def read_prediction(path, expected_ids):
    need(Path(path).stat().st_size <= 8*1024*1024, 'prediction exceeds 8 MiB')
    with zipfile.ZipFile(path) as z:
        need(len(z.namelist())==len(set(z.namelist())), 'duplicate ZIP members')
        need(sum(x.file_size for x in z.infolist()) <= 8*1024*1024, 'expanded prediction exceeds 8 MiB')
    with np.load(path, allow_pickle=False) as z:
        ids=z['ids']
        need(ids.ndim==1 and ids.dtype.kind in 'US', 'ids must be strings')
        ids=ids.tolist()
        need(len(ids)==len(set(ids)), 'duplicate ids')
        need(len(ids)==len(expected_ids) and set(ids)==set(expected_ids), 'id set mismatch')
        keys={'ids'}
        for i in range(len(ids)):
            keys.update({f'edges_{i}',f'gauge_{i}'})
            keys.update(f'tensor_{i}_{j}' for j in range(22))
        need(set(z.files)==keys, 'prediction keys mismatch')
        records={}
        for i,name in enumerate(ids):
            edges=z[f'edges_{i}']; gauge=z[f'gauge_{i}']; tensors=[z[f'tensor_{i}_{j}'] for j in range(22)]
            need(edges.shape==(22,3) and edges.dtype.kind in 'iu', 'edges must be int[22,3]')
            need(np.all((edges>=0)&(edges<=44)), 'edge labels outside 0..44')
            for a,rank in [(gauge,2)]+[(t,3) for t in tensors]:
                need(a.dtype.kind in 'fc' and a.ndim==rank, 'tensor dtype/rank')
                need(all(1<=d<=64 for d in a.shape), 'invalid tensor dimensions')
                need(np.isfinite(a).all(), 'nonfinite tensor')
            records[name]=(edges,tensors,gauge)
    return records


def topology(edges,tensors,gauge):
    occ=defaultdict(list)
    for i,(es,t) in enumerate(zip(edges,tensors)):
        need(len(set(es))==3, 'repeated edge on a node')
        for axis,e in enumerate(es):occ[int(e)].append((i,axis,t.shape[axis]))
    need(set(occ)==set(range(45)), 'must have 24 physical and 21 auxiliary edges')
    adjacency={i:set() for i in range(46)}
    for e,items in occ.items():
        if e<24:
            need(len(items)==1 and items[0][2]==2, 'physical leg mismatch')
            a,b=e,items[0][0]+24
        else:
            need(len(items)==2 and items[0][2]==items[1][2], 'bond endpoints mismatch')
            need(1<=items[0][2]<=4, 'auxiliary bond dimension exceeds 4')
            a,b=items[0][0]+24,items[1][0]+24
        need(a!=b and b not in adjacency[a], 'self-loop or parallel edge')
        adjacency[a].add(b);adjacency[b].add(a)
    seen=set(); todo=[0]
    while todo:
        n=todo.pop()
        if n not in seen:seen.add(n);todo.extend(adjacency[n]-seen)
    need(len(seen)==46 and sum(map(len,adjacency.values()))==90,'network must be a connected tree')
    need(all(len(adjacency[n])==(1 if n<24 else 3) for n in adjacency),'binary-tree degree violation')
    centers=[int(e) for e,c in Counter(edges[:,2]).items() if c==2]
    need(len(centers)==1 and centers[0]>=24,'one central auxiliary edge required')
    roots=[i for i,es in enumerate(edges) if es[2]==centers[0]]
    need(gauge.shape==(tensors[roots[0]].shape[2],tensors[roots[1]].shape[2]),'gauge dimensions mismatch')
    return occ,roots


def dense_fidelity(record,target):
    edges,tensors,gauge=record
    occ,roots=topology(edges,tensors,gauge)
    visited=set()
    def branch(i):
        need(i not in visited,'cycle in oriented tree');visited.add(i)
        parts=[]
        for e in edges[i,:2]:
            if e<24:parts.append((np.eye(2),[int(e)]))
            else:
                child=[j for j,axis,dim in occ[int(e)] if j!=i]
                need(len(child)==1 and edges[child[0],2]==e,'orient axes 0,1 away from the center')
                parts.append(branch(child[0]))
        (a,la),(b,lb)=parts
        t=np.einsum('ai,bj,ijk->abk',a,b,tensors[i],optimize=True)
        return t.reshape(a.shape[0]*b.shape[0],tensors[i].shape[2]),la+lb
    a,la=branch(roots[0]);b,lb=branch(roots[1])
    need(visited==set(range(22)) and sorted(la+lb)==list(range(24)),'missing physical axes')
    dense=((a@gauge)@b.T).reshape((2,)*24)
    dense=np.ascontiguousarray(dense.transpose(np.argsort(la+lb))).reshape(-1)
    norm2=float(np.vdot(dense,dense).real)
    target=np.asarray(target).reshape(-1)
    target=target/np.linalg.norm(target)
    need(np.isfinite(norm2) and norm2>0,'zero/nonfinite reconstructed norm')
    f=float(abs(np.vdot(target,dense))**2/norm2)
    need(np.isfinite(f) and 0<=f<=1+1e-12,'invalid fidelity')
    return min(f,1.0),max(dim for e,items in occ.items() if e>=24 for _,_,dim in items)

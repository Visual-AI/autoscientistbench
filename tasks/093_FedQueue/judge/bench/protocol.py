"""Frozen protocol constants and independent APPFL partition-index replay."""
import numpy as np

SCHEDULER={'same_init_model':True,'t_sync':10.0,'lr_base':.003,'warm_up_steps':10,
 'alpha_queue':.5,'alpha_compute':.5,'safety_buffer':2.0,'admission_tolerance':.2,
 'simulate_queue':True,'q_init':2.0,'alpha_admission':.2,'beta_queue':.5,'theta':2.0,
 'queue_mode':'lognormal','queue_means':'1.5,2.5,3.5,4.5','queue_sigma':.9,
 'slowdown':'1.0,1.0,1.0,1.0','log_fedqueue_metrics':True}

def partition_indices(y,seed):
    # Arithmetic and RNG order match APPFL dirichlet_noniid_partition exactly.
    rng=np.random.RandomState(seed)
    labels=sorted(set(int(v) for v in y))
    indices={v:np.flatnonzero(y==v).tolist() for v in labels}
    for v in labels:rng.shuffle(indices[v])
    p1=[1/4 for _ in range(4)]
    p2=[len(indices[v]) for v in labels];p2=[p/sum(p2) for p in p2]
    weights=rng.dirichlet([8*i for i in p1])
    individuals=rng.dirichlet([.1*i for i in p2],4)
    portions=np.zeros(individuals.shape)
    for i in range(4):
        for j in range(len(labels)):
            denom=np.dot(weights,individuals.transpose()[j])
            portions[i][j]=weights[i]*individuals[i][j]/denom if denom>0 else 0.0
    samples=[len(indices[v]) for v in labels]
    matrix=np.multiply(np.array([samples]*4),portions).transpose()
    for i in range(len(samples)):
        total=0
        for j in range(3):matrix[i][j]=int(matrix[i][j]);total+=matrix[i][j]
        matrix[i][3]=samples[i]-total
    ends=np.cumsum(np.array(matrix.transpose(),dtype=np.int32),axis=0)
    return {f'client_{i}':np.asarray([idx for j,v in enumerate(labels) for idx in indices[v][0 if i==0 else ends[i-1][j]:ends[i][j]]],dtype=np.int64) for i in range(4)}

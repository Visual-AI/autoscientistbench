"""Counted public oracle API. Private state access or replacement is forbidden."""
import importlib.util
from pathlib import Path
import numpy as np
import scipy.linalg
_spec = importlib.util.spec_from_file_location('_frozen_objective', Path(__file__).with_name('_objective.py'))
_obj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_obj)
METHODS = ('exact_func','inexact_func','exact_grad_alpha1','exact_grad_alpha0')

def make_oracle(A,b,instance_id,method):
    raw = _obj.create_log_sum_exp_zero_oracle(A.T,b,1.0)
    events=[]
    records=[]
    counts={'grad':0,'inverse':0}
    class API:
        def func(self,x):
            events.append('func')
            return raw.func(x)
        def grad(self,x):
            counts['grad']+=1
            events.append('grad')
            return raw.grad(x)
        def hess(self,x):
            events.append('hess')
            return raw.hess(x)
        def approx_hess(self,x):
            events.append('approx_hess')
            raw._update_a_and_pi(x)
            return raw.matmat_ATsA(raw.pi)/raw.mu
        def solve(self,matrix,rhs):
            # Exactly the authors' successful cho_solve counter convention.
            result=scipy.linalg.cho_solve(scipy.linalg.cho_factor(matrix,lower=False),rhs)
            counts['inverse']+=1
            events.append('inverse')
            return result
        def record(self,x):
            rec={'x':np.asarray(x).tolist(),'func':float(raw.func(x)),
                 'grad_calls':counts['grad'],'matrix_inverses':counts['inverse'],
                 'events':list(events)}
            records.append(rec)
            events.clear()
            return rec
        def history(self):
            return list(records)
    return API()

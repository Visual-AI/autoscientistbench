"""Deterministic authors' PA ODE and MF stationary solver; never reads MC data."""
import ctypes, pathlib, time, os
import numpy as np
from scipy.integrate import solve_ivp
HERE=pathlib.Path(__file__).resolve().parent
lib=ctypes.CDLL(str(pathlib.Path(os.environ['PA_LIBRARY'])))
ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
lib.rhs.argtypes=[ptr,ptr]+[ctypes.c_double]*4
lib.rhs.restype=None

def rhs(y,k,p,alpha=.1,q=3):
    out=np.empty(10); lib.rhs(np.ascontiguousarray(y),out,k,p,alpha,q); return out

def curve(k,pgrid=np.arange(51)/100,rtol=1e-7,atol=1e-9,alpha=.1):
    start=time.process_time(); wall=time.perf_counter(); rows=[]
    for p in pgrid:
        y0=np.zeros(10); y0[0]=1
        sol=solve_ivp(lambda t,y:rhs(y,k,p,alpha),(0.,2000.),y0,method='RK45',rtol=rtol,atol=atol,t_eval=[2000.])
        assert sol.success
        y=sol.y[:,-1]; cuu=y[0]+y[1]+y[2]+y[3]; cud=y[1]+y[4]+y[5]+y[6]
        cdu=y[2]+y[5]+y[7]+y[8]
        mass=y[[0,4,7,9]].sum()+2*y[[1,2,3,5,6,8]].sum()
        rows.append([p,cuu+cud,cuu+cdu,cud+cdu,np.max(np.abs(rhs(y,k,p,alpha))),abs(mass-1),sol.nfev])
    return np.array(rows),{'cpu_s':time.process_time()-start,'wall_s':time.perf_counter()-wall,'rtol':rtol,'atol':atol,'end_time':2000,'solver':'scipy RK45 (Dormand-Prince, as MATLAB ode45)'}

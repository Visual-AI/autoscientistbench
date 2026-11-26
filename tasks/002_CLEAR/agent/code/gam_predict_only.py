"""PIRLS for prediction-only evaluation bootstrap models.

Preserves all iterations and convergence logic. Reassociates the coefficient
matrix product to avoid constructing an m-by-n matrix at every iteration, and
omits post-fit covariance/GCV statistics never consumed by this table.
"""
import inspect
import textwrap
import pygam.pygam as pg
import gam_linalg
ORIGINAL=gam_linalg.ORIGINAL

def install(directory=None):
    gam_linalg.DIRECTORY=directory
    source=textwrap.dedent(inspect.getsource(ORIGINAL))
    old='np.linalg.svd(np.vstack([R, E]))'
    assert source.count(old)==1
    source=source.replace(old,'_opt30_svd(np.vstack([R, E]))')
    old='B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)\n        coef_new = B.dot(pseudo_data).flatten()'
    assert source.count(old)==1
    source=source.replace(old,'coef_new = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T.dot(pseudo_data)).flatten()')
    old='self._estimate_model_statistics(\n        Y, modelmat, inner=None, BW=WB.T, B=B, weights=weights, U1=U1\n    )'
    assert source.count(old)==1
    source=source.replace(old,'pass  # unused post-fit covariance, GCV, AIC and standard errors')
    scope=dict(ORIGINAL.__globals__,_opt30_svd=gam_linalg.economy_svd)
    exec(compile(source,__file__,'exec'),scope)
    pg.GAM._pirls=scope['_pirls']

def restore():pg.GAM._pirls=ORIGINAL

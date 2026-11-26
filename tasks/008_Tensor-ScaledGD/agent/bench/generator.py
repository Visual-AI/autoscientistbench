"""Frozen generation utilities for trusted materialization and public development.
Evaluation workers receive observations only; private generation seeds and truth
never enter submitted code. The family arithmetic is unchanged.
"""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_frozen_family',HERE/'family/cpu_opt/protocol.py')
_family=importlib.util.module_from_spec(spec);spec.loader.exec_module(_family)
PANELS=[('TRPCA_noiseless','DFT'),('TRPCA_noiseless','DCT'),('TC_noiseless','DFT'),('TC_noiseless','DCT'),('RTC','DFT'),('RTC','DCT'),('TR','DFT'),('TR','DCT')]
KAPPAS=[1,5,10,20]

def truth_script(base,transform,out,seed):
    """Operator data materialization only; forbidden to algorithms."""
    s=_family.generate(base,transform,out)
    prefix, cases=s.split('for i_kappa = 1:length(kappa_list)',1)
    body=cases.split('    %% Spectral initialization')[0]
    # X_star precedes observations; avoid constructing the huge regression operator.
    if base=='TR':prefix=prefix.split('As=zeros(',1)[0]
    body=body.split('    S_supp_idx')[0] if '    S_supp_idx' in body else body
    body=body.split('    Omega =')[0] if '    Omega =' in body else body
    body=body.split("    y=As'")[0] if "    y=As'" in body else body
    # Corruption RNG is irrelevant to X_star, whose U/V seeds precede all kappa draws.
    s=prefix+'for i_kappa=1:length(kappa_list)\n'+body+f"\nsave('-mat7-binary',sprintf('{out}/truth_%d.mat',i_kappa),'X_star');\nend\n"
    return s.replace('20261022',str(int(seed)))

def instance_prefix(base,transform,out,index,seed):
    """Trusted builder for an instance; never called with private seeds by workers.
    Returned source must be executed unmodified. Private truth is oracle-only.
    """
    s=_family.generate(base,transform,out,index=index)
    s=s.split('    %% Spectral initialization')[0]
    s=s.replace('20261022',str(int(seed)))
    return s


def regression_initialization(out,seed):
    """Trusted builder: shared dense operator and original paired spectral initialization."""
    import sys
    old=sys.modules.get('protocol')
    sys.modules['protocol']=_family
    try:
        spec=importlib.util.spec_from_file_location('_frozen_batch',HERE/'family/protocol30.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.regression(out).split('for t=1:T',1)[0].replace('20261022',str(int(seed)))
    finally:
        if old is not None:sys.modules['protocol']=old
        else:sys.modules.pop('protocol',None)

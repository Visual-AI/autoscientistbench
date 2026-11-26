"""Frozen numerical checks; declared evidence still requires model replay and patch review."""
import math
import re
import numpy as np

POST = dict(optim='SGD',momentum=0,weight_decay=0.0005,lr_schedule='onecycle',max_lr=0.1,post_unlearn_clip=False)
MODEL_CONFIG = dict(name='tiny_net_cifar', n_classes=10)
PARAM_SHAPES = {
    'Conv_0': {'bias': (32,), 'kernel': (3, 3, 3, 32)},
    'Conv_1': {'bias': (64,), 'kernel': (3, 3, 32, 64)},
    'Dense_0': {'bias': (10,), 'kernel': (64, 10)},
}
NOISE_SHAPES = [list(shape) for layer in PARAM_SHAPES.values() for shape in layer.values()]

def verify_model_params(params):
    if not isinstance(params, dict) or set(params) != set(PARAM_SHAPES):
        raise ValueError('requires the fixed CIFAR10TinykNet parameter tree')
    for layer, shapes in PARAM_SHAPES.items():
        if not isinstance(params[layer], dict) or set(params[layer]) != set(shapes):
            raise ValueError('changed research-model parameter tree')
        for name, shape in shapes.items():
            array = np.asarray(params[layer][name])
            if array.shape != shape or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError('invalid fixed-model parameter tensor: ' + layer + '/' + name)
    return True

BASE_SHA='d7bad9cf74356aa49905251fbaa2d695d8435a04874c5d7aa316a8831c04efd6'

def verify(c):
    if c['format']!='pabi-certificate-v1':raise ValueError('certificate format')
    if c['base_sha256']!=BASE_SHA:raise ValueError('wrong frozen starting model')
    cfg=c['config'];u=cfg['unlearning'];records=c['noise']
    if cfg.get('model') != MODEL_CONFIG:raise ValueError('research object is the supplied CIFAR10TinykNet')
    if u.get('optim') != 'SGD' or u.get('momentum', 0) != 0:raise ValueError('PABI certificate requires the verified clipped-SGD transition')
    if u['epochs']!=50:raise ValueError('requires 50 measured epochs')
    if not isinstance(cfg['dataset']['batch_size'],int) or cfg['dataset']['batch_size']<1:
        raise ValueError('invalid training batch size')
    if not isinstance(c['steps_per_epoch'],int) or c['steps_per_epoch']<1 or not isinstance(c['final_step'],int) or c['final_step']<0:
        raise ValueError('invalid diagnostic training counts')
    if len(c['epoch_sha256'])!=50 or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in c['epoch_sha256']):
        raise ValueError('missing or malformed epoch hashes')
    if u['algorithm']!='iteration' or u['noise_schedule']!='constant' or u['lr_schedule']!='constant' or u['init_model_clip_type']!='clip' or u['init_sigma']!=0:raise ValueError('unsupported certification setting')
    if any(r.get('noise_shapes') != NOISE_SHAPES for r in records):raise ValueError('noise must cover the fixed research-model parameters')
    if not records or [r['step'] for r in records]!=list(range(1,len(records)+1)):raise ValueError('noise steps')
    if len({tuple(r['input_key']) for r in records})!=len(records) or len({r['noise_sha256'] for r in records})!=len(records):raise ValueError('reused noise/key')
    if any(a['output_key']!=b['input_key'] for a,b in zip(records,records[1:])):raise ValueError('broken key chain')
    eta=float(u['max_lr']);lam=float(u['weight_decay']);clip=float(u['iteration']['grad_clip']);C0=float(u['init_model_clip']);delta=float(u['delta'])
    if not 0<delta<=1e-5 or not 0<eta*lam<1 or clip<=0 or C0<=0:raise ValueError('invalid privacy parameters')
    rho=1-eta*lam;T=len(records);common=rho**T
    factor=(2*C0*common+2*clip/lam*(1-common))**2*eta*lam*(2-eta*lam)/(2*(1-common**2))
    sigmas=np.array([r['sigma'] for r in records],dtype=float)
    if not np.isfinite(sigmas).all() or np.any(sigmas<=0) or not np.allclose(sigmas,sigmas[0],rtol=0,atol=0):raise ValueError('invalid sigma')
    if any(not math.isclose(r['eta'],eta,rel_tol=1e-6) for r in records):raise ValueError('eta mismatch')
    renyi=factor/float(sigmas[0])**2
    alpha=np.linspace(1.001,1000,10000)
    eps=float(np.min(renyi-(np.log(delta)+np.log(alpha))/(alpha-1)+np.log((alpha-1)/alpha)))
    if not math.isfinite(eps) or eps>1.004:raise ValueError('epsilon exceeds 1.004')
    return dict(epsilon=eps,delta=delta,noise_steps=T,renyi=renyi)

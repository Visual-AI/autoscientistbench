"""NA-QSP benchmark, paper Eqs. 20,21,26,29,30; notebook cells 1,3,4,5.
Tensor order is oscillator,qubit, as in the notebook's GCR cell.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import qutip as qt
from qutip.qip.operations import rx

ALPHA = 2.0
THETA = np.pi / 2
BUDGET = np.pi / 4
CONVERGENCE_ATOL = 1e-9
INPUTS = ('plus_alpha', 'minus_alpha', 'even_cat')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('All gate parameters must be finite real numbers')
    return float(x)

def axis(v):
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError('Axis must be a list of three real numbers')
    a = np.array([number(x) for x in v])
    if abs(np.linalg.norm(a) - 1) > 1e-12:
        raise ValueError('Axis must have unit norm (absolute tolerance 1e-12)')
    return a / np.linalg.norm(a)

def validate(gates):
    if not isinstance(gates, list):
        raise ValueError('Pulse must be a JSON list of gates in execution order')
    amplitudes = []
    for g in gates:
        if not isinstance(g, dict):
            raise ValueError('Each gate must be an object')
        kind = g.get('type')
        keys = {'qubit_rot': {'type','axis','angle'},
                'cd': {'type','lam','phi','qubit_axis'}}
        if kind not in keys or set(g) != keys[kind]:
            raise ValueError('Unsupported gate type or fields')
        if kind == 'qubit_rot':
            axis(g['axis']); number(g['angle'])
        else:
            axis(g['qubit_axis']); number(g['phi'])
            amplitudes.append(abs(number(g['lam'])))
    total = math.fsum(amplitudes)
    if not math.isfinite(total):
        raise ValueError('Nonfinite total CD amplitude')
    return total

def baselines(alpha=ALPHA, theta=THETA):
    # Eq. 30 acts right to left; JSON gates act in listed order.
    c = theta / (2 * alpha)
    gcr = [dict(type='cd', lam=c, phi=np.pi/2, qubit_axis=[0,1,0]),
           dict(type='cd', lam=c, phi=0, qubit_axis=[1,0,0])]
    phi1 = np.arccos(-theta / (4*np.pi))
    bb1 = [dict(type='cd', lam=c, phi=0, qubit_axis=[1,0,0])]
    for angle, lam in [(phi1,np.pi/(2*alpha)), (3*phi1,np.pi/alpha),
                       (phi1,np.pi/(2*alpha))]:
        bb1.append(dict(type='cd', lam=lam, phi=0,
                        qubit_axis=[float(np.cos(angle)),float(np.sin(angle)),0]))
    return {'gcr': gcr, 'bb1': bb1}

def operators(n):
    # Verbatim definitions in NA-QSP_sims.ipynb cell 3.
    aOp = qt.destroy(n)
    aOp1 = aOp.dag()
    xOp = (aOp + aOp1)/2
    pOp = (-1j)*(aOp-aOp1)/2
    return xOp, pOp

def states(n, alpha=ALPHA, theta=THETA):
    g = qt.basis(2,0)  # cell 1
    cp, cm = qt.coherent(n,alpha), qt.coherent(n,-alpha)
    qp, qm = rx(theta)*g, rx(-theta)*g
    tp, tm = qt.tensor(cp,qp), qt.tensor(cm,qm)
    initial = [qt.tensor(cp,g), qt.tensor(cm,g), qt.tensor((cp+cm).unit(),g)]
    target = [tp, tm, (tp+tm).unit()]
    return initial, target, (qp,qm)

def simulate(gates, n, alpha=ALPHA, theta=THETA):
    validate(gates)
    x,p = operators(n)
    out, _, _ = states(n, alpha, theta)
    paulis = [qt.sigmax(),qt.sigmay(),qt.sigmaz()]
    for gate in gates:
        if gate['type'] == 'cd':
            s = sum(v*m for v,m in zip(axis(gate['qubit_axis']),paulis))
            q = np.cos(gate['phi'])*x + np.sin(gate['phi'])*p
            U = (-1j*gate['lam']*qt.tensor(q,s)).expm()
        else:
            s = sum(v*m for v,m in zip(axis(gate['axis']),paulis))
            U = qt.tensor(qt.qeye(n),(-1j*gate['angle']/2*s).expm())
        out = [U*v for v in out]
    return np.column_stack([v.full().ravel() for v in out])

def measure(output,n,alpha=ALPHA,theta=THETA):
    _,target,(qp,qm) = states(n,alpha,theta)
    out = [qt.Qobj(output[:,i],dims=[[n,2],[1,1]]) for i in range(3)]
    norm_errors = [abs(v.norm()-1) for v in out]
    if max(norm_errors)>1e-10:
        raise ValueError('Simulation lost norm')
    # Notebook cells 4/5: normalize output and compute 1-|state.overlap(state2)|^2.
    out = [v.unit() for v in out]
    inf = {k: float(np.clip(1-abs(t.overlap(v))**2,0,1))
           for k,t,v in zip(INPUTS,target,out)}
    pe = {}
    for k,q,v in zip(INPUTS[:2],(qp,qm),out[:2]):
        pe[k] = float(np.clip(1-qt.expect(qt.tensor(qt.qeye(n),qt.ket2dm(q)),v),0,1))
    # Eq.20 diagnostic extension to an entangled cat: joint sign(x)/sigma_y
    # readout. A single qubit projector cannot distinguish cat success/failure.
    vals,vecs = np.linalg.eigh(operators(n)[0].full())
    projplus = qt.Qobj(vecs[:,vals>0] @ vecs[:,vals>0].conj().T)
    success = qt.tensor(projplus,qt.ket2dm(qp)) + qt.tensor(qt.qeye(n)-projplus,qt.ket2dm(qm))
    pe['even_cat'] = float(np.clip(1-qt.expect(success,out[2]),0,1))
    return {'infidelity':inf, 'pe':pe, 'max_norm_error':max(norm_errors)}

def evaluate(gates, outputs=None, alpha=ALPHA, theta=THETA):
    total = validate(gates)
    if outputs is None:
        outputs = {n:simulate(gates,n,alpha,theta) for n in (400,600)}
    m = {n:measure(outputs[n],n,alpha,theta) for n in (400,600)}
    delta = max(abs(m[400]['infidelity'][k]-m[600]['infidelity'][k]) for k in INPUTS)
    within = total <= BUDGET
    converged = delta <= CONVERGENCE_ATOL
    worst = max(m[400]['infidelity'].values())
    return dict(infidelity_by_input=m[400]['infidelity'], worst_infidelity=worst,
                pe_by_input=m[400]['pe'], total_cd_amplitude=total,
                infidelity_ncav600=m[600]['infidelity'], pe_ncav600=m[600]['pe'],
                within_budget=within, gate_set_valid=True, ncav600_check_passed=converged,
                convergence_max_abs_difference=delta, convergence_atol=CONVERGENCE_ATOL,
                cd_amplitude_budget=BUDGET, score=worst if within and converged else 1.0,
                parameters=dict(alpha=alpha, theta_t=theta, Delta=1, Ncav=400, check_Ncav=600),
                std_error=0.0, std_error_by_input={k:0.0 for k in INPUTS},
                pe_std_error_by_input={k:0.0 for k in INPUTS},
                uncertainty='Deterministic state-vector simulation; no sampling or Monte Carlo. SE=0 denotes no sampling error, not zero numerical error.',
                definition='1-FH=1-|<normalized target|normalized output>|^2; Eq.21, notebook cells 4/5. Branch targets rx(+-pi/2)|g>; cat target normalized coherent sum, Eq.26.',
                pe_definition='Branches: notebook qubit-projector failure (Eq.20). Cat: joint sign(x)/target sigma_y failure, a diagnostic extension of Eq.20; not scored.',
                max_state_norm_error=max(m[n]['max_norm_error'] for n in m))

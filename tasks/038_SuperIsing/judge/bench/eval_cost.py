"""Sec. 8 pp.26–27, Table 2 and Fig.12: relative RMS, free energy scale."""
import csv
import json
import math
from pathlib import Path
import sys

OPERATORS = {
    0: [('sigma', .58444), ('epsilon', 1.58444), ('box_sigma', 2.58444), ('sigma_prime', 2.8869)],
    .5: [('chi', 1.08444), ('gamma_d_chi', 2.08444), ('box_chi', 3.08444)],
    1: [('d_sigma', 1.58444), ('d_epsilon', 2.58444)],
    1.5: [('d_chi', 2.08444), ('G', 2.5), ('gamma_dd_chi', 3.08444)],
    2: [('dd_sigma', 2.58444), ('T', 3.0)],
}
FIG13 = dict(zip(['sigma','epsilon','box_sigma','chi','gamma_d_chi','box_chi','d_sigma','d_epsilon','d_chi','gamma_dd_chi','dd_sigma','G','T','sigma_prime'], [.61,1.50,2.53,1.06,2.06,3.11,1.46,2.58,2.08,3.14,2.57,2.51,2.98,2.76]))

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def cost(path):
    with open(path) as f:
        states = [{k: float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    assert len(states) == 40
    assert all(sum(r['lz2'] == k for r in states) == 20 for k in (0,1))
    states.sort(key=lambda r:r['E'])
    for r in states:
        r['l'] = round((math.sqrt(1+4*r['L2'])-1)) / 2
        r['L2_error'] = abs(r['L2']-r['l']*(r['l']+1))
    assert all(r['L2_error'] < 1e-5 for r in states), 'Non-integral spin assignment'
    assert all(r['residual'] < 1e-6 for r in states), 'Eigensolver residual exceeds 1e-6'
    assert states[0]['l'] == 0
    e0 = states[0]['E']
    groups = {l: [r for r in states[1:] if r['l'] == l] for l in OPERATORS}
    sigma_gap = groups[0][0]['E'] - e0
    assigned = []
    for l, refs in OPERATORS.items():
        assert len(groups[l]) >= len(refs), (path,l,len(groups[l]))
        for r,(name,ref) in zip(groups[l],refs):
            assigned.append(dict(operator=name,l=l,reference=ref,E=r['E'],
                                 unscaled_delta=.58444*(r['E']-e0)/sigma_gap,
                                 L2_error=r['L2_error'], residual=r['residual']))
    ratios = [r['unscaled_delta']/r['reference'] for r in assigned]
    scale = sum(ratios)/sum(x*x for x in ratios)
    for r in assigned:
        r['delta'] = scale*r['unscaled_delta']
        r['relative_deviation'] = r['delta']/r['reference']-1
        r['fig13_delta'] = FIG13[r['operator']]
        r['difference_from_fig13'] = r['delta']-r['fig13_delta']
        r['mc_std_error'] = 0.0
    value = math.sqrt(sum(r['relative_deviation']**2 for r in assigned)/14)
    return dict(cost=value, lambda_opt=scale, E0=e0, sigma_gap=sigma_gap,
                operators=assigned, states=40, operator_count=14,
                max_L2_error=max(r['L2_error'] for r in states),
                max_eigen_residual=max(r['residual'] for r in states),
                std_error=0.0, uncertainty='Deterministic ED; no Monte Carlo sampling. Solver seed variability is not estimated.')

def evaluate(out):
    manifest = json.loads((out/'simulation.json').read_text())
    cases = []
    for item in manifest['cases']:
        measured = cost(out/item['file'])
        assert abs(measured['cost']-item['training_cost']) < 1e-12
        cases.append({**item, **measured})
    best = min((r for r in cases if r['nmf']==9), key=lambda r:r['cost'])
    # No three numerical scoring constraints were supplied in the task. These are
    # explicit reproducibility checks, not invented paper accuracy thresholds.
    constraints = [
      dict(metric='unchanged model and fixed couplings (Sec.8, Table2)', threshold='t=1.5,U=0.25,g=1; authors Hamiltonian', value=manifest['fixed_couplings'], satisfied=manifest['fixed_couplings']==[1.5,.25,1.0]),
      dict(metric='spectrum and operator coverage (Fig.12)', threshold='N=9; 20 states per sector; all 14 assigned',value=dict(nmf=best['nmf'],states=best['states'],operators=best['operator_count']),satisfied=best['states']==40 and best['operator_count']==14),
      dict(metric='calibration and optimisation protocol (Sec.8)', threshold='analytic optimal lambda; <=40 local ED evaluations; both parameter triples and N=6,7,8 evaluated',value=dict(local_ed_evaluations=manifest['local_ed_evaluations'],lambda_opt=best['lambda_opt']),satisfied=manifest['local_ed_evaluations']<=40 and all(any(r['tag']==tag for r in cases) for tag in ['table6','table7','table8','table9','example9']))]
    result = dict(experiment='Super-Ising SCFT, Sec.8, Table2, Fig.12–13',
        task_metric='relative RMS scaling-dimension cost at N_mf=9',value=best['cost'],
        score=None, protocol_validated_cost=best['cost'] if all(c['satisfied'] for c in constraints) else 1.0,
        std_error=0.0, paper_value=.025, theoretical_bound=.0005,
        best_tag=best['tag'],best_parameters=best['parameters'],lambda_opt=best['lambda_opt'],
        constraints=constraints, constraint_note='The prompt refers to three constraints but does not enumerate their definitions or thresholds. These three protocol checks are reported; an external constraint-gated benchmark score cannot be certified without its definitions.',
        cases=cases, reductions=manifest['reductions'], method_unchanged=True,
        definition='Sec.8 pp.26–27, Fig.12: sqrt(mean((Delta/Delta_ref-1)^2)); Delta=lambda*0.58444*(E-E0)/(E_sigma-E0); lambda=sum(r)/sum(r^2).',
        operator_list_corrections=[], uncertainty='MC standard errors are zero because this is deterministic ED, not a Monte Carlo estimator. No seed replication; solver variability is unestimated; residuals are supplied.')
    atomic_json(out/'metric.json',result)
    atomic_json(out/'metrics.json',result)
    print(f"N=9 baseline cost: {best['cost']:.10f}; paper representative value: 0.025 (stated range 0.02–0.03)")
    print(f"Best: {best['tag']}, parameters={best['parameters']}, lambda={best['lambda_opt']:.12f}")
    for r in cases:
        if not r['tag'].startswith('local'):
            print(f"N={r['nmf']} {r['tag']}: cost={r['cost']:.10f}")
    for c in constraints:
        print(f"Protocol constraint {c['metric']}: {c['value']}; required {c['threshold']}; pass={c['satisfied']}")
    print(result['constraint_note'])
    for r in best['operators']:
        print(f"{r['operator']:15s} delta={r['delta']:.8f} reference={r['reference']:.5f} deviation={r['relative_deviation']:+.4%} Fig13={r['fig13_delta']:.2f}")
    print(f"Maximum L2 assignment error={best['max_L2_error']:.3e}; eigenpair residual={best['max_eigen_residual']:.3e}")

if __name__=='__main__':
    evaluate(Path(sys.argv[1]))

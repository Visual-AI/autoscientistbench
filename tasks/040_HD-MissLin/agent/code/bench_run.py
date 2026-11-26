#!/usr/bin/env python3
"""Per-instance fitting driver; data loading, process scheduling and seed protocol only."""
import argparse, concurrent.futures, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from repro.method import ScaledLasso, DualCD, DualObj, DebiasProg_SCSfirst


def fit_nuisance(record):
    iid, X_sim, Y_sim, R2, x, seed = record
    n,d=X_sim.shape
    # Independent algorithm stream. No state or seed of the data generator enters prediction.
    digest=hashlib.sha256(X_sim.tobytes()).digest()
    instance_seed=(int.from_bytes(digest[:4], 'little') + int(seed)) % 2**32
    np.random.seed(instance_seed)
    # --- Lasso pilot estimator on the MAR-observed data (authors: lam0='univ') ---
    beta_pilot2, sigma_pilot2 = ScaledLasso(X=X_sim[R2 == 1, :], Y=Y_sim[R2 == 1], lam0='univ')

    # --- Propensity score: L1-regularised logistic regression with CV (authors' grid) ---
    zeta2 = np.logspace(-1, np.log10(300), 40) * np.sqrt(np.log(d) / n)
    lr2 = LogisticRegressionCV(Cs=1 / zeta2, cv=5, penalty='l1', scoring='neg_log_loss',
                               solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R2)
    prop_score2 = lr2.predict_proba(X_sim)[:, 1]
    return beta_pilot2, sigma_pilot2, prop_score2


def cv_cell(X_sim, x, prop_score2, train_ind, test_ind, g):
    # One (fold, gamma) cell: cold-started and draws no random numbers, so any worker can run it.
    X_train = X_sim[train_ind, :]
    X_test = X_sim[test_ind, :]
    prop_score2_train = prop_score2[train_ind]
    prop_score2_test = prop_score2[test_ind]
    w_train2 = DebiasProg_SCSfirst(X=X_train.copy(), x=x,
                                   Pi=np.diag(prop_score2_train), gamma_n=g)
    if any(np.isnan(w_train2)):
        return np.nan
    ll_train2 = DualCD(X=X_train, x=x, Pi=np.diag(prop_score2_train), gamma_n=g,
                       ll_init=None, eps=1e-8, max_iter=5000)
    if sum(abs(w_train2 + np.dot(X_train, ll_train2)
               / (2 * np.sqrt(X_train.shape[0]))) > 1e-3) > 0:
        return np.nan
    return float(np.squeeze(
        DualObj(X_test, x=x, Pi=np.diag(prop_score2_test),
                ll_cur=ll_train2, gamma_n=g)))


def fit_rule(record, beta_pilot2, sigma_pilot2, prop_score2, gamma_n2_opt):
    iid, X_sim, Y_sim, R2, x, seed = record
    n=X_sim.shape[0]
    w_obs2 = DebiasProg_SCSfirst(X=X_sim.copy(), x=x, Pi=np.diag(prop_score2),
                                 gamma_n=gamma_n2_opt)
    ll_obs2 = DualCD(X=X_sim, x=x, Pi=np.diag(prop_score2), gamma_n=gamma_n2_opt,
                     ll_init=None, eps=1e-9)
    m_deb2 = np.dot(x, beta_pilot2) + np.sum(w_obs2 * R2 * (Y_sim - np.dot(X_sim, beta_pilot2))) / np.sqrt(n)
    asym_var2 = np.sqrt(np.sum(prop_score2 * w_obs2 ** 2) / n)
    sigma_hat2 = sigma_pilot2
    return {"m_deb": float(m_deb2), "asym_se": float(asym_var2),
            "sigma_hat": float(sigma_hat2), "gamma_n": float(gamma_n2_opt)}


def fit_instance(record, pool):
    iid, X_sim, Y_sim, R2, x, seed = record
    t_start=time.time()
    # Pilot and propensity draw from the global RNG, so they stay one serial task per instance.
    beta_pilot2, sigma_pilot2, prop_score2 = pool.submit(fit_nuisance, record).result()

    # --- Tuning-parameter CV over the authors' grid (MAR branch only) ---
    gamma_n_lst = np.linspace(0.001, np.max(abs(x)), 41)
    cv_fold = 5
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=0)
    cells = [[pool.submit(cv_cell, X_sim, x, prop_score2, train_ind, test_ind, g) for g in gamma_n_lst]
             for train_ind, test_ind in kf.split(X_sim)]
    dual_loss2 = np.array([[c.result() for c in fold] for fold in cells])

    mean_dual_loss2 = np.mean(dual_loss2, axis=0)
    std_dual_loss2 = np.std(dual_loss2, axis=0, ddof=1) / np.sqrt(cv_fold)

    # --- Three tuning rules (verbatim from the authors' script) ---
    out = {}
    para_rule = ['1se', 'mincv', 'minfeas']
    for rule in para_rule:
        if rule == 'mincv':
            gamma_n2_opt = gamma_n_lst[np.nanargmin(mean_dual_loss2)]
        if rule == '1se':
            One_SE2 = (mean_dual_loss2 > np.nanmin(mean_dual_loss2)
                       + std_dual_loss2[np.nanargmin(mean_dual_loss2)]) & \
                (gamma_n_lst < gamma_n_lst[np.nanargmin(mean_dual_loss2)])
            if sum(One_SE2) == 0:
                One_SE2 = np.full((len(gamma_n_lst),), True)
            gamma_n_lst2 = gamma_n_lst[One_SE2]
            gamma_n2_opt = gamma_n_lst2[np.nanargmin(mean_dual_loss2[One_SE2])]
        if rule == 'minfeas':
            gamma_n2_opt = np.min(gamma_n_lst[~np.isnan(mean_dual_loss2)])
        out[rule] = pool.submit(fit_rule, record, beta_pilot2, sigma_pilot2, prop_score2, gamma_n2_opt)
    out = {rule: f.result() for rule, f in out.items()}

    elapsed=time.time()-t_start
    print(json.dumps({"id":iid,"elapsed_s":round(elapsed,3)}),flush=True)
    return iid,out['1se']


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train');tr.add_argument('--out',required=True)
    pr=sub.add_parser('predict');pr.add_argument('--model',required=True);pr.add_argument('--input',required=True);pr.add_argument('--output',required=True)
    args=ap.parse_args();seed=int(os.environ.get('BENCH_SEED','1'))
    if args.mode=='train':
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        cfg={"seed":seed,"pilot":"scaled_lasso_univ","propensity_cv":5,"propensity_grid":40,
             "debias_cv":5,"debias_grid":41,"rule":"1se","solver_order":["SCS","ECOS","CVXOPT"]}
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        return
    cfg=json.loads((Path(args.model)/'config.json').read_text())
    if int(cfg['seed'])!=seed:raise ValueError('model and BENCH_SEED mismatch')
    with np.load(args.input,allow_pickle=False) as z:
        records=[(str(i),X,y,R,x,seed) for i,X,y,R,x in zip(z['id'],z['X'],z['y'],z['R'],z['query'])]
    # One pool of BENCH_CPUS workers serves all instances' tasks; each instance has a waiting driver thread.
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('BENCH_CPUS','1'))) as pool:
        pool.submit(int).result()  # forks the workers before any driver thread exists
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(records)) as drivers:
            results=list(drivers.map(lambda record:fit_instance(record,pool),records))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','m_deb','sigma_hat','asym_se'])
        for iid,r in results:w.writerow([iid,r['m_deb'],r['sigma_hat'],r['asym_se']])
if __name__=='__main__':main()

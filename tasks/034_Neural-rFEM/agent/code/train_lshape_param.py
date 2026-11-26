#!/usr/bin/env python3
"""Protocol adapter of the reproduced parametric L-shape training driver.
The model, FEM loss, Adam update (including its constant step index), sampling
algorithm and full training length are preserved. Input rows are materialized.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, help='path to code/2D/L_shape/parametric')
    p.add_argument('--data-dir', default=None, help='defaults to <repo>/Dataset')
    p.add_argument('--n', type=int, default=10, help='elements per side (paper: N=10)')
    p.add_argument('--steps', type=int, default=1_400_000, help='Adam steps (paper: 1.4e6)')
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--lr-schedule', choices=['fixed', 'notebook'], default='fixed',
                   help="'fixed': constant LR (paper Sec. 4.3.2 for N=10); 'notebook': the "
                        "official notebook's schedule (drop to 1e-3 after step 1400)")
    p.add_argument('--impl', choices=['repo', 'fast', 'fast_cb'], default='fast',
                   help="'repo': official path; 'fast_cb': identical assembly with the scipy "
                        "solve behind a pure_callback; 'fast' (default): solve on the host "
                        "between two jitted calls - same arithmetic, no callback overhead")
    p.add_argument('--log-every', type=int, default=2000)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=42, help='PRNGKey(42) as in the notebook')
    p.add_argument('--sample-seed', type=int, default=100, help='PRNGKey(100) as in the notebook')
    p.add_argument('--max-seconds', type=float, default=0.0, help='stop early (pilot safety)')
    return p.parse_args()


def main():
    args = parse_args()
    repo = os.path.abspath(args.repo)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # the official modules import each other through relative paths
    os.chdir(repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)                       # notebook's own directory
    sys.path.insert(0, os.path.abspath('../..'))   # code/2D          -> CSR_functions
    sys.path.insert(0, os.path.abspath('..'))      # code/2D/L_shape  -> integration_info

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.nn as nn
    from jax import random, grad
    from jax.example_libraries import optimizers
    import csv
    import fem_system
    from fem_system import solve_and_loss, solve_and_loss_scipy

    t_start = time.time()
    print(f"[train] jax {jax.__version__} | devices {jax.devices()} | N={args.n} steps={args.steps} "
          f"lr={args.lr} schedule={args.lr_schedule} impl={args.impl}", flush=True)

    # Materialized training rows only; validation does not enter optimization.
    with open(os.path.join(args.data_dir, 'train', 'input.csv'), newline='') as f:
        train_data = np.array([[float(r['s1']), float(r['s2'])]
                               for r in csv.DictReader(f)], dtype=np.float64)
    print(f"[train] {len(train_data)} training tuples", flush=True)

    def key_of(alpha):
        return '|'.join(repr(float(v)) for v in np.asarray(alpha).ravel())

    # ---------------- grid ----------------------------------------------------
    N = args.n
    n_nodes = 2 * (N - 1)
    uniform_grid = jnp.concatenate([jnp.linspace(0, 1, n_nodes // 2 + 1),
                                    jnp.linspace(0, 1, n_nodes // 2 + 1)])

    # uniform-mesh Ritz energy for every tuple (notebook cell 5)
    t0 = time.time()
    ritz_dict_train = {key_of(a): float(solve_and_loss(uniform_grid, a))
                       for a in train_data}
    print(f"[train] uniform-mesh Ritz energies: {len(ritz_dict_train)} solves in {time.time()-t0:.1f}s", flush=True)

    # ---------------- neural network (notebook cell 8) -----------------------
    def init_params(layers, key):
        Ws, bs = [], []
        for i in range(len(layers) - 1):
            std_lecun = jnp.sqrt(1 / layers[i])
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i], layers[i + 1])) * std_lecun)
            bs.append(jnp.zeros(layers[i + 1]))
        return [Ws, bs]

    @jax.jit
    def forward_pass(A_t, params):
        Ws, bs = params
        H = A_t
        for i in range(len(Ws) - 1):
            H = jnp.matmul(H, Ws[i]) + bs[i]
            H = jnp.tanh(H)
        nodes_ = jnp.squeeze(jnp.array([jnp.matmul(H, Ws[-1]) + bs[-1]]))
        n_n = nodes_.size
        thetax = nn.softmax(nodes_[:n_n // 2])
        thetay = nn.softmax(nodes_[n_n // 2:])
        nodesx = jnp.cumsum(thetax)
        nodesx = jnp.insert(jnp.insert(nodesx, 0, 0.5), 0, 0).sort()
        nodesy = jnp.cumsum(thetay)
        nodesy = jnp.insert(jnp.insert(nodesy, 0, 0.5), 0, 0).sort()
        return jnp.concatenate([nodesx, nodesy])

    layers = [2, 10, 10, n_nodes]
    params = init_params(layers, random.PRNGKey(args.seed))

    def _compute_loss(K_batch, u_batch, F_batch):
        Ku = jax.experimental.sparse.bcsr_dot_general(
            K_batch, u_batch, dimension_numbers=(([2], [1]), ([0], [0])))
        return 0.5 * jnp.sum(u_batch * Ku, axis=1) - jnp.sum(F_batch * u_batch, axis=1)

    # ---------------- loss, gradients and equivalence check ------------------
    verify = {}
    if args.impl in ('fast', 'fast_cb'):
        import fast_fem
        import scipy.sparse as sp
        # vmapped assembler: (nodes, sigma[1,2]) -> (BCSR K, F) like build_system_vmap
        assemble_batch, assemble_per_sigma, solve_host, pattern = \
            fast_fem.make_assembler(fem_system, jax, jnp, N)
        col_np, indptr_np, nnz, n_grid = pattern
        print(f"[train] fast implementation: BCSR pattern nnz={nnz} n={n_grid}", flush=True)

        _assemble_jit = jax.jit(lambda p, al: assemble_batch(forward_pass(al[0], p), al))

        def host_solve(K_batch, F_batch):
            """scipy sparse direct solve on one system (host side, constant wrt params)"""
            m = sp.csr_matrix((np.asarray(K_batch)[0], col_np, indptr_np), shape=(n_grid, n_grid))
            return np.asarray(sp.linalg.spsolve(m, np.asarray(F_batch)[0]))[None, :]

        def solve_batch(params, alphas):
            """Ritz energies for a batch of sigma on the r-adaptive mesh (not jitted).
            Batch size is 1 in the training loop, so the shared mesh is exact."""
            K, Fb = assemble_batch(forward_pass(alphas[0], params), alphas)
            u = (solve_host(K.data, Fb) if args.impl == 'fast_cb'
                 else jnp.asarray(host_solve(K.data, Fb)))
            return _compute_loss(K, u, Fb)

        def solve_any(nodes, sig2):
            K, Fb = assemble_batch(nodes, sig2)
            u = (solve_host(K.data, Fb) if args.impl == 'fast_cb'
                 else jnp.asarray(host_solve(K.data, Fb)))
            return float(_compute_loss(K, u, Fb)[0])

        def loss(params, alphas):
            losses = solve_batch(params, alphas)
            return jnp.array([losses[i] / abs(ritz_dict_train[key_of(alpha)])
                              for i, alpha in enumerate(alphas)]).mean()

        def raw_loss(params, alphas):
            """mean Ritz energy over the batch (notebook loss before the constant
            1/|J_uniform| normalisation); differentiates through the callback path"""
            K, Fb = assemble_batch(forward_pass(alphas[0], params), alphas)
            u = solve_host(K.data, Fb)
            return _compute_loss(K, u, Fb).mean()

        def loss_u(params, alphas, u):
            """same loss with the FEM solution u held constant: the official code
            does not differentiate the linear solver either (solve_using_scipy_bwd
            returns zeros)."""
            K, Fb = assemble_batch(forward_pass(alphas[0], params), alphas)
            return _compute_loss(K, u, Fb).mean()

        def _scale_grad(g, nrm):
            return [[W / nrm for W in g[0]], [b / nrm for b in g[1]]]

        _adam_jit_cb = jax.jit(lambda p, st, al, lr_, nrm:
                               optimizers.adam(lr_)[1](
                                   0, _scale_grad(grad(raw_loss)(p, al), nrm), st))

        @jax.jit
        def _adam_jit_u(params, opt_state, sigma_batch, lr_, nrm, u):
            g = grad(loss_u)(params, sigma_batch, u)
            return optimizers.adam(lr_)[1](0, _scale_grad(g, nrm), opt_state)

        def adam_step(params, opt_state, sigma_batch, lr_, nrm):
            if args.impl == 'fast_cb':
                return _adam_jit_cb(params, opt_state, sigma_batch, lr_, nrm)
            K, Fb = _assemble_jit(params, sigma_batch)          # jitted assembly
            u = jnp.asarray(host_solve(K.data, Fb))             # host solve
            return _adam_jit_u(params, opt_state, sigma_batch, lr_, nrm, u)

        # ---- numerical equivalence with the official path -------------------
        verify['assembly'] = fast_fem.verify(fem_system, jax, jnp, assemble_batch, forward_pass,
                                             params, [[0.1, 0.1], [1.0, 3.0], [10.0, 0.1]])
        ref_loss = lambda p, a: solve_and_loss_scipy(forward_pass(a[0], p), a[0].reshape(-1, 2)).mean()
        checks = []
        for sv in [train_data[0], train_data[139], train_data[-1]]:
            sv = np.asarray(sv, dtype=np.float64)
            al = jnp.asarray(sv).reshape(1, 2)
            gh, gk = grad(raw_loss)(params, al), grad(ref_loss)(params, al)
            checks.append(dict(sigma=list(sv),
                               loss_diff=abs(float(raw_loss(params, al)) - float(ref_loss(params, al))),
                               grad_max_diff=max(float(jnp.max(jnp.abs(x - y)))
                                                 for x, y in zip(gh[0] + gh[1], gk[0] + gk[1]))))
        verify['loss_grad'] = checks
        print('[train] equivalence vs official path: ' + json.dumps(verify), flush=True)
    else:
        def solve_any(nodes, sig2):
            return float(solve_and_loss_scipy(nodes, sig2).item())

        def loss(params, alphas):
            losses = solve_and_loss_scipy(forward_pass(alphas[0], params), alphas[0].reshape(-1, 2))
            return jnp.array([losses[i] / abs(ritz_dict_train[key_of(alpha)])
                              for i, alpha in enumerate(alphas)]).mean()

        grad_loss = grad(loss)
        adam_step = None

    opt_init, opt_update, get_params = optimizers.adam(args.lr)
    opt_state = opt_init(params)

    train_alphas = jnp.array(train_data)

    # ---------------- training loop (notebook cell 10) -----------------------
    key = random.PRNGKey(args.sample_seed)
    if args.impl in ('fast', 'fast_cb'):
        # identical PRNG stream to the notebook's per-step
        #   key, subkey = random.split(key); idx = random.choice(subkey, n_alphas, (1,), replace=False)
        # computed once in a compiled loop instead of one dispatch per step.
        def _index_stream(key0, n_steps, n_alphas):
            def body(carry, _):
                key = carry
                key, sub = random.split(key)
                idx = random.choice(sub, n_alphas, (1,), replace=False).squeeze()
                return key, idx
            _, idxs = jax.lax.scan(body, key0, None, length=n_steps)
            return idxs

        t_ = time.time()
        idx_seq = np.asarray(jax.jit(_index_stream, static_argnums=(1, 2))(
            key, args.steps + 1, train_alphas.shape[0]))
        print(f"[train] sigma-index stream ({args.steps+1} draws) in {time.time()-t_:.1f}s", flush=True)
        nrm_seq = np.array([abs(ritz_dict_train[key_of(a)]) for a in train_data])
        train_data_np = np.asarray(train_data, dtype=np.float64)
    history = []
    t0 = time.time()
    t_last = t0
    lr = args.lr
    key, subkey = random.split(key)
    sigma_batch = train_alphas[random.choice(subkey, train_alphas.shape[0], (1,), replace=False)]
    steps_done = 0
    prof = {} if os.environ.get('PHY09_PROFILE') else None
    def _tick(name, t_ref):
        if prof is not None:
            prof[name] = prof.get(name, 0.0) + (time.perf_counter() - t_ref)
        return time.perf_counter()
    for step in range(args.steps + 1):
        if step > 0:
            _t = time.perf_counter()
            if args.impl in ('fast', 'fast_cb'):
                i = int(idx_seq[step])
                sigma_batch = jnp.asarray(train_data_np[i:i + 1])
                nrm = nrm_seq[i]
            else:
                key, subkey = random.split(key)
                idx = random.choice(subkey, train_alphas.shape[0], (1,), replace=False)
                sigma_batch = train_alphas[idx]
                nrm = abs(ritz_dict_train[key_of(sigma_batch)])
            params = get_params(opt_state)
            _t = _tick('sample', _t)
            if adam_step is not None:
                opt_state = adam_step(params, opt_state, sigma_batch, lr, nrm)
            else:
                opt_state = opt_update(0, grad_loss(params, sigma_batch), opt_state)
            _t = _tick('step', _t)
            if args.lr_schedule == 'notebook' and step == 1400:
                lr = 1e-3
                opt_init, opt_update, get_params = optimizers.adam(lr)
                opt_state = opt_init(get_params(opt_state))
            steps_done = step
        if step % args.log_every == 0:
            _t = time.perf_counter()
            params = get_params(opt_state)
            train_loss = float(loss(params, sigma_batch))
            now = time.time()
            rate = 1000 * (now - t0) / max(1, step) if step else float('nan')
            print(f"[train] step {step:>9d}  train_loss {train_loss:.6e}  "
                  f"lr {lr:.1e}  ({now - t0:.0f}s, "
                  f"{rate:.3f} ms/step)", flush=True)
            t_last = now
            _tick('monitor', _t)
            history.append(dict(step=step, train_loss=train_loss, lr=lr))
            with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
                json.dump(dict(args=vars(args), steps_done=steps_done, history=history), f, indent=1)
        if args.max_seconds and step > 0 and time.time() - t0 > args.max_seconds:
            print(f"[train] stopping after {step} steps (--max-seconds {args.max_seconds})", flush=True)
            break

    train_wall = time.time() - t0
    params = get_params(opt_state)

    flat = {}
    Ws, bs = params
    for i, W in enumerate(Ws):
        flat[f'W{i}'] = np.asarray(W)
    for i, b in enumerate(bs):
        flat[f'b{i}'] = np.asarray(b)
    np.savez(os.path.join(out_dir, 'trained_params.npz'), **flat)

    if prof is not None:
        print('[train] profile (s): ' + json.dumps({k: round(v, 2) for k, v in prof.items()}), flush=True)
    meta = dict(args=vars(args), steps_done=steps_done, train_wall_s=train_wall, profile=prof,
                ms_per_step=1000 * train_wall / max(1, steps_done),
                verify=verify,
                jax=jax.__version__, jax_device=str(jax.devices()[0]),
                total_wall_s=time.time() - t_start)
    with open(os.path.join(out_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"[train] finished {steps_done} steps in {train_wall:.1f}s "
          f"({meta['ms_per_step']:.3f} ms/step); outputs in {out_dir}", flush=True)


if __name__ == '__main__':
    main()

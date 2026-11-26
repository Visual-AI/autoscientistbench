#!/usr/bin/env python3
"""Performance-only variant of the official FEM assembly, used by the training
driver so that the paper's 1.4e6 Adam steps fit in the time budget.

`code/2D/L_shape/parametric/fem_system.py` rebuilds the sparse structure (COO ->
CSR/BCSR: argsort, dedup, segment_sum over ~2.6e3 entries) inside *every*
training step, and solves the linear system through a scipy callback.  The
sparsity pattern, however, depends only on the mesh topology, which never
changes: the network only moves the nodes' coordinates.  This module therefore

  1. builds the pattern once from the official code (create_COO / to_Bcsr) and
     replaces the per-step COO->BCSR conversion by one scatter-add onto the
     fixed pattern  (a linear map, identical values up to summation order), and
  2. drives the same scipy sparse solver through a jax.pure_callback with the
     solver treated as non-differentiable, exactly like the repo's
     custom_vjp `solve_using_scipy`.

Everything else (element_stiffness, element_load, Dirichlet labelling, load
vector scatter-adds) is the official code, unchanged.  `verify()` checks the
result against the official path.
"""
import numpy as np
import scipy.sparse as sp


def make_assembler(F, jax, jnp, N):
    """Return assemble_fast(nodes, sigma) -> (K_bcsr, F), equal to
    F.assemble_CSR(..., )+F.apply_boundary_conditions(...) of the official code."""
    sparse = F.sparse

    nx = N + 1                     # mesh nodes per side
    n_grid = nx * nx               # global node count
    # topology only: mesh built from a dummy uniform grid
    x0 = jnp.linspace(0, 1, nx)
    coords0, elements = F.generate_mesh(nx, nx, x0, x0)
    n_elem = elements.shape[0]

    # ---- reproduce the COO->BCSR map of the official to_Bcsr, once ----------
    zero_ke = jnp.zeros((n_elem, 16))
    coo0 = F.create_COO(elements, zero_ke, n_grid)
    row, col = coo0.row, coo0.col
    max_col = col.max() + 1
    keys = row * max_col + col
    sort_idx = jnp.argsort(keys)
    sorted_keys = keys[sort_idx]
    unique_mask = jnp.diff(jnp.concatenate([jnp.array([-1]), sorted_keys])) != 0
    to_remove = 16 + 2 * (nx - 2) * 6 + 2 * (nx - 2) * 6 + 9 * (nx - 2) * (nx - 2)   # as in assemble_CSR
    n_removes = 16 * n_elem - to_remove
    unique_indices = jnp.nonzero(unique_mask, size=sorted_keys.shape[0] - n_removes)[0]
    inverse_indices = jnp.cumsum(unique_mask) - 1
    nnz = int(unique_indices.shape[0])
    col_idx = jnp.asarray(coo0.col)[sort_idx][unique_indices].astype(jnp.int32)
    sorted_row = jnp.asarray(row)[sort_idx][unique_indices]
    change_mask = jnp.concatenate([jnp.array([True]), sorted_row[1:] != sorted_row[:-1]])
    indices_filas = jnp.nonzero(change_mask, size=sorted_row.size, fill_value=0)[0]
    indptr = jnp.append(indices_filas[0:n_grid], nnz).astype(jnp.int32)
    sort_idx = jnp.asarray(sort_idx)
    inverse_indices = jnp.asarray(inverse_indices).astype(jnp.int32)

    # CSR of the fixed pattern, for the host solver
    col_np = np.asarray(col_idx, dtype=np.int32)
    indptr_np = np.asarray(indptr, dtype=np.int32)

    def assemble_fast(nodes, sigma):
        nx_ = nodes.shape[0] // 2
        x = nodes[:nx_]
        y = nodes[nx_:]
        node_coords, elements_ = F.generate_mesh(nx_, nx_, x, y)

        mask_dirichlet = jax.vmap(F.is_zero_quadrant, in_axes=(0, 0))(node_coords[:, 0], node_coords[:, 1])
        mask_dirichlet = mask_dirichlet.at[0:nx_].set(1)
        mask_dirichlet = mask_dirichlet.at[nx_ * (nx_ - 1) + 0:nx_ * (nx_ - 1) + nx_].set(1)
        mask_dirichlet = mask_dirichlet.at[0:nx_ * (nx_ - 1) + 1:nx_].set(1)
        mask_dirichlet = mask_dirichlet.at[nx_ - 1:nx_ * nx_:nx_].set(1)
        dirichlet_nodes = jnp.where(mask_dirichlet == 1, size=nx_ * nx_, fill_value=nx_ * nx_)[0]

        dir_elements = jax.vmap(lambda el: F.element_dirichlet(el, dirichlet_nodes))(elements_)
        dir_values = jnp.zeros(elements_.shape, dtype=jnp.float64)

        coords_corners = jnp.concatenate([node_coords[elements_[:, 0]], node_coords[elements_[:, 2]]], axis=1)
        fe_values = jax.vmap(F.element_load)(coords_corners)
        ke_values, dir_cumsum_values = jax.vmap(
            lambda coords, d, v: F.element_stiffness(coords, d, v, sigma),
            in_axes=(0, 0, 0))(coords_corners, dir_elements, dir_values)

        # fixed linear map: sorted COO values -> BCSR data (was to_Bcsr)
        flat = ke_values.reshape(-1)[sort_idx]
        Kdata = jnp.zeros(nnz, dtype=flat.dtype).at[inverse_indices].add(flat)
        K = sparse.BCSR((Kdata, col_idx, indptr), shape=(n_grid, n_grid))

        # identical order of scatter-adds as the official assemble_CSR
        Fv = jnp.zeros(n_grid)
        for k in range(4):
            Fv = Fv.at[elements_[:, k]].add(fe_values[:, k])
        for k in range(4):
            Fv = Fv.at[elements_[:, k]].add(-dir_cumsum_values[:, k])
        K, Fv = F.apply_boundary_conditions(K, Fv, dirichlet_nodes)
        return K, Fv

    def solve_host_factory():
        def _np_solve(kdata, fvec):
            out = []
            for i in range(kdata.shape[0]):
                m = sp.csr_matrix((np.asarray(kdata[i]), col_np, indptr_np), shape=(n_grid, n_grid))
                out.append(sp.linalg.spsolve(m, np.asarray(fvec[i])))
            return np.stack(out, axis=0)

        @jax.custom_vjp
        def solve_host(Kdata, Fv):
            out_shape = jax.ShapeDtypeStruct((Kdata.shape[0], n_grid), Kdata.dtype)
            return jax.pure_callback(_np_solve, out_shape, Kdata, Fv)

        def _fwd(Kdata, Fv):
            return solve_host(Kdata, Fv), None

        def _bwd(res, g):
            return (None, None)

        solve_host.defvjp(_fwd, _bwd)
        return solve_host

    # batched versions: (nodes, sigma[B,2]) like build_system_vmap (one mesh for the
    # whole batch), and (nodes[B,:], sigma[B,2]) with one mesh per sigma
    assemble_batch = jax.vmap(assemble_fast, in_axes=(None, 0))
    assemble_per_sigma = jax.vmap(assemble_fast, in_axes=(0, 0))

    class Assembler(tuple):
        """(assemble_batch, assemble_per_sigma, solve_host, pattern)"""

    return (assemble_batch, assemble_per_sigma, solve_host_factory(),
            (col_np, indptr_np, nnz, n_grid))


def verify(F, jax, jnp, assemble_fast, forward_pass, params, sigma_vecs, assert_atol=1e-11):
    """Compare the fast assembly against the official build_system_vmap."""
    import numpy as _np
    report = []
    for sv in sigma_vecs:
        sv = jnp.asarray(_np.asarray(sv, dtype=_np.float64))
        nodes = forward_pass(sv, params)
        K_f, F_f = assemble_fast(nodes, sv.reshape(-1, 2))
        _c, K_r, F_r = F.build_system_vmap(nodes, sv.reshape(-1, 2))
        dK = float(jnp.max(jnp.abs(K_r.data[0] - K_f.data)))
        dF = float(jnp.max(jnp.abs(F_r[0] - F_f)))
        report.append((_np.asarray(sv).ravel().tolist(), dK, dF))
        assert dK < assert_atol and dF < assert_atol, (dK, dF)
    return report

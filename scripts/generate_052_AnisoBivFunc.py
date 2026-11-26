#!/usr/bin/env python3
"""Deterministic local observation shard generator for 052_AnisoBivFunc.

Materializes the bit-exact float64 observation shards for 052_AnisoBivFunc
(`val` = 76.8 GB, `test` = 76.8 GB) locally using the frozen R generator kernels
(`agent/bench/generator_source/{dgp.R,fast_kernels.R}`), the portable generation
plan (`judge/generation/plan.csv`), and the task's snapshot R runtime
(`runtime/r.tar`, fetched from Hugging Face).

Why local generation instead of hosting 153.6 GB on Hugging Face?
052_AnisoBivFunc consists of 24,000 pure synthetic bivariate functional surfaces
(120 shards per split, 76,812,000,000 bytes each). Generating them locally with
the exact frozen R runtime and fixed seeds reproduces the exact bytes in ~5-10
minutes on 12 CPU cores without transferring 153.6 GB over the network.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# CPU-ISA-independent exact R materializer:
# Uses the task's frozen dgp.R (fbm_fft) and fast_kernels.R (nn_index) with R's
# original crossprod() coordinate projection (matching the original AVX-512/LDOUBLE
# rounding bit-for-bit on all x86_64 CPUs, including midpoint ties at alpha = pi/4)
# and precomputes the 10 (side, alpha) lookup tables once before forking.
_EXACT_GENERATOR_R = r"""
a <- commandArgs(TRUE)
bench_dir <- a[1]
plan_file <- a[2]
cores <- as.integer(a[3])
source(file.path(bench_dir, "generator_source", "dgp.R"))
source(file.path(bench_dir, "generator_source", "fast_kernels.R"))

.grid_cache <- new.env(parent = emptyenv())
get_grid_indices <- function(t_n, e_n, alpha) {
  key <- sprintf("%d_%d_%.17g", t_n, e_n, alpha)
  if (exists(key, envir = .grid_cache, inherits = FALSE)) {
    return(get(key, envir = .grid_cache, inherits = FALSE))
  }
  t1_tilde <- seq(0, 1, length.out = e_n)
  t2_tilde <- seq(0, 1, length.out = e_n)
  t1 <- seq(0, abs(cos(alpha)) + sin(alpha), length.out = t_n)
  if (alpha <= pi/2) {
    t2 <- seq(-sin(alpha), cos(alpha), length.out = t_n)
    neg <- t2[t2 < 0]
    neg_idx <- if (length(neg) == 0L) integer(0) else nn_index(-neg, t1)
    t1_neg_idx <- integer(0)
  } else {
    t1_proj <- seq(cos(alpha), sin(alpha), length.out = t_n)
    t2 <- seq(cos(alpha) - sin(alpha), 0, length.out = t_n)
    t1_neg_idx <- nn_index(-t1_proj[t1_proj < 0], t1)
    neg_idx <- nn_index(-t2[t2 < 0], t1)
  }
  M <- unname(as.matrix(expand.grid(t2_tilde = t2_tilde, t1_tilde = t1_tilde)))
  u1 <- c(cos(alpha), sin(alpha))
  u2 <- c(-sin(alpha), cos(alpha))
  t1_u1 <- apply(M, 1, function(x) crossprod(x, u1))
  t2_u2 <- apply(M, 1, function(x) crossprod(x, u2))
  val <- list(
    t1_len = length(t1),
    t1_max = max(t1),
    t1_neg_idx = t1_neg_idx,
    neg_idx = neg_idx,
    i1 = nn_index(t1_u1, t1),
    i2 = nn_index(t2_u2, t2)
  )
  assign(key, val, envir = .grid_cache)
  val
}

fbm_sheet_exact <- function(t_n, e_n, alpha, H1, H2, type = "sum", sigma = NULL) {
  if (H1 <= H2) alpha <- alpha + pi/2
  while (alpha >= pi) alpha <- alpha - pi
  g <- get_grid_indices(t_n, e_n, alpha)
  if (alpha <= pi/2) {
    B1 <- fbm_fft(H = H1, n = g$t1_len, grid_max = g$t1_max)
    B2_tilde <- fbm_fft(H = H2, n = g$t1_len, grid_max = g$t1_max)
    if (length(g$neg_idx) == 0L) {
      B2_minus <- c()
    } else {
      B2_minus <- -B2_tilde[g$neg_idx]
    }
    B2_plus <- B2_tilde[seq_len(g$t1_len - length(B2_minus))]
    B2 <- c(B2_minus, B2_plus)
  } else {
    B1_tilde <- fbm_fft(H = H1, n = g$t1_len, grid_max = g$t1_max)
    B1_minus <- -B1_tilde[g$t1_neg_idx]
    B1_plus <- B1_tilde[seq_len(g$t1_len - length(B1_minus))]
    B1 <- c(B1_minus, B1_plus)
    B2_tilde <- fbm_fft(H = H2, n = g$t1_len, grid_max = g$t1_max)
    B2_minus <- -B2_tilde[g$neg_idx]
    B2_plus <- B2_tilde[seq_len(g$t1_len - length(B2_minus))]
    B2 <- c(B2_minus, B2_plus)
  }
  X <- if (type == "sum") B1[g$i1] + B2[g$i2] else B1[g$i1] * B2[g$i2]
  if (missing(sigma) || is.null(sigma)) {
    X
  } else {
    X + rnorm(n = length(X), mean = 0, sd = sigma)
  }
}

plan <- read.csv(plan_file, stringsAsFactors = FALSE)
uniq <- unique(plan[, c("side", "alpha"), drop = FALSE])
for (u in seq_len(nrow(uniq))) get_grid_indices(uniq$side[u], uniq$side[u], uniq$alpha[u])

groups <- split(seq_len(nrow(plan)), plan$path)
t0 <- Sys.time()
res <- parallel::mclapply(groups, function(ix) {
  rows <- plan[ix, , drop = FALSE]; target <- rows$path[1]
  expected <- sum(rows$N * rows$side^2 * 8)
  if (file.exists(target) && file.info(target)$size == expected) return(TRUE)
  output <- paste0(target, ".partial")
  f <- file(output, "wb")
  tryCatch({
    for (k in seq_len(nrow(rows))) {
      row <- rows[k, ]; set.seed(row$seed)
      for (i in seq_len(row$N)) {
        X <- fbm_sheet_exact(row$side, row$side, row$alpha, 0.8, 0.5, "sum", row$sigma)
        writeBin(as.double(X), f, size = 8, endian = "little")
      }
    }
    close(f)
    stopifnot(file.info(output)$size == expected)
    stopifnot(file.rename(output, target))
    cat(sprintf("completed %s %d records %.0f bytes %.1fs\n", basename(target), nrow(rows), expected, as.numeric(difftime(Sys.time(), t0, units = "secs")))); flush.console()
    TRUE
  }, error = function(e) { try(close(f), silent = TRUE); stop(e) })
}, mc.cores = cores, mc.preschedule = FALSE)
stopifnot(all(vapply(res, isTRUE, logical(1))))
cat(sprintf("FULL_GENERATION_COMPLETE records=%d seconds=%.1f\n", nrow(plan), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
"""


def load_plan(plan_path: Path, splits: set[str], shard_names: set[str] | None = None, limit_shards: int | None = None):
    rows_by_key = defaultdict(list)
    ordered_keys = []
    with plan_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row["split"] not in splits:
                continue
            if shard_names and row["file"] not in shard_names:
                continue
            key = (row["split"], row["file"])
            if key not in rows_by_key:
                ordered_keys.append(key)
            rows_by_key[key].append(row)
    if limit_shards is not None:
        ordered_keys = ordered_keys[:limit_shards]
    selected = []
    expected_sizes = {}
    for key in ordered_keys:
        srows = rows_by_key[key]
        selected.extend(srows)
        rel_path = srows[0]["path"]
        expected_sizes[rel_path] = sum(int(r["N"]) * int(r["side"]) * int(r["side"]) * 8 for r in srows)
    return fieldnames, selected, expected_sizes


def verify_shards(task_dir: Path, expected_sizes: dict[str, int]) -> tuple[list[str], list[str]]:
    ok_list, bad_list = [], []
    for rel_path, exp_bytes in sorted(expected_sizes.items()):
        p = task_dir / rel_path
        if p.is_file() and p.stat().st_size == exp_bytes:
            ok_list.append(rel_path)
        else:
            actual = p.stat().st_size if p.is_file() else -1
            bad_list.append(f"{rel_path} (expected {exp_bytes}, got {actual})")
    return ok_list, bad_list


def generate_shards(
    task_dir: Path,
    split: str = "all",
    cores: int | None = None,
    shard_names: list[str] | None = None,
    limit_shards: int | None = None,
    verify_only: bool = False,
    use_system_r: bool = False,
    no_sandbox: bool = False,
) -> dict:
    task_dir = Path(task_dir).resolve()
    plan_path = task_dir / "judge" / "generation" / "plan.csv"
    gen_r = task_dir / "agent" / "bench" / "generator.R"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing 052_AnisoBivFunc plan: {plan_path}")
    if not gen_r.is_file():
        raise FileNotFoundError(f"Missing 052_AnisoBivFunc generator: {gen_r}")

    splits = {"val", "test"} if split == "all" else {split}
    fieldnames, rows, expected_sizes = load_plan(
        plan_path,
        splits=splits,
        shard_names=set(shard_names) if shard_names else None,
        limit_shards=limit_shards,
    )
    if not rows:
        raise ValueError(f"No rows selected from {plan_path} for split={split!r}")

    ok_before, bad_before = verify_shards(task_dir, expected_sizes)
    total_bytes = sum(expected_sizes.values())
    if verify_only:
        return {
            "ok": len(bad_before) == 0,
            "verified": len(ok_before),
            "missing_or_mismatch": bad_before,
            "total_shards": len(expected_sizes),
            "total_bytes": total_bytes,
        }
    if not bad_before:
        print(f"[052_AnisoBivFunc] All {len(ok_before)} shard(s) ({total_bytes / 1e9:.2f} GB) already present and size-verified.")
        return {"ok": True, "generated": 0, "verified": len(ok_before), "total_bytes": total_bytes}

    cores = int(cores or min(os.cpu_count() or 4, 12))
    print(
        f"[052_AnisoBivFunc] Generating {len(bad_before)}/{len(expected_sizes)} missing shard(s) "
        f"(split={split}, total target {total_bytes / 1e9:.2f} GB, cores={cores})...",
        flush=True,
    )

    # Ensure output directories exist
    for rel_path in expected_sizes:
        (task_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aniso-gen-") as tmp:
        tmp_dir = Path(tmp)
        resolved_plan = tmp_dir / "plan_resolved.csv"
        runner_r = tmp_dir / "generate_exact.R"
        runner_r.write_text(_EXACT_GENERATOR_R, encoding="utf-8")
        bench_dir = task_dir / "agent" / "bench"

        if use_system_r:
            rscript = shutil.which("Rscript")
            if not rscript:
                raise RuntimeError("--system-r requested, but Rscript not found on PATH")
            with resolved_plan.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    rr = dict(r)
                    rr["path"] = str((task_dir / r["path"]).resolve())
                    w.writerow(rr)
            env = dict(os.environ)
            env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            subprocess.run([rscript, str(runner_r), str(bench_dir), str(resolved_plan), str(cores)], env=env, check=True)
        else:
            from bench.runtime import bind_runtime, materialize, runtime_env
            from bench.sandbox import Sandbox
            from bench.task import load_task

            task = load_task(task_dir)
            rt_tar = task_dir / "runtime" / "r.tar"
            if not rt_tar.is_file():
                raise FileNotFoundError(
                    f"Missing {rt_tar}. Fetch the 052_AnisoBivFunc R snapshot runtime first:\n"
                    f"  python3 -m bench fetch 052_AnisoBivFunc --only runtime"
                )
            materialize(task)

            if no_sandbox:
                with resolved_plan.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in rows:
                        rr = dict(r)
                        rr["path"] = str((task_dir / r["path"]).resolve())
                        w.writerow(rr)
                env = dict(os.environ)
                env.update(runtime_env(task, host=True))
                env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                rscript = shutil.which("Rscript", path=env["PATH"])
                if not rscript:
                    raise RuntimeError(f"Rscript not found in runtime PATH={env['PATH']}")
                subprocess.run([rscript, str(runner_r), str(bench_dir), str(resolved_plan), str(cores)], env=env, check=True)
            else:
                # Run Rscript inside bwrap with exact virtual runtime mounts so compiled R prefixes work on any Linux host
                in_task = "/ws/task052"
                in_plan = "/tmp/plan_resolved.csv"
                in_runner = "/tmp/generate_exact.R"
                with resolved_plan.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in rows:
                        rr = dict(r)
                        rr["path"] = f"{in_task}/{r['path']}"
                        w.writerow(rr)

                sb = Sandbox(unshare_net=True, own_uid=False)
                sb.ro_bind(bench_dir, f"{in_task}/agent/bench")
                if "val" in splits:
                    sb.rw_bind(task_dir / "agent" / "data" / "val" / "input", f"{in_task}/agent/data/val/input")
                if "test" in splits:
                    sb.rw_bind(task_dir / "judge" / "test" / "input", f"{in_task}/judge/test/input")
                sb.ro_file(resolved_plan, in_plan)
                sb.ro_file(runner_r, in_runner)
                sb.tmpfs("/home/gen").rw_bind(tmp_dir, "/tmp/work")
                bind_runtime(sb, task)
                renv = runtime_env(task, host=False)
                renv.update({
                    "HOME": "/home/gen",
                    "TMPDIR": "/tmp/work",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "LANG": "C.UTF-8",
                })
                sb.setenv(**renv)
                sb.cwd = "/tmp/work"
                argv = sb.argv(["Rscript", in_runner, f"{in_task}/agent/bench", in_plan, str(cores)], tmp_dir / "sb")
                subprocess.run(argv, check=True)

    ok_after, bad_after = verify_shards(task_dir, expected_sizes)
    if bad_after:
        raise RuntimeError(f"052_AnisoBivFunc verification failed for {len(bad_after)} shard(s): {bad_after[:5]}")
    print(f"[052_AnisoBivFunc] Verified {len(ok_after)}/{len(expected_sizes)} shard(s) ({total_bytes / 1e9:.2f} GB) OK.")
    return {"ok": True, "generated": len(bad_before), "verified": len(ok_after), "total_bytes": total_bytes}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministically generate 052_AnisoBivFunc observation shards locally.")
    ap.add_argument("--task-dir", default=str(REPO_ROOT / "tasks" / "052_AnisoBivFunc"), help="Path to tasks/052_AnisoBivFunc")
    ap.add_argument("--split", choices=["val", "test", "all"], default="all", help="Which split(s) to generate")
    ap.add_argument("--cores", type=int, default=None, help="Worker processes for R parallel::mclapply")
    ap.add_argument("--shard", action="append", dest="shards", help="Specific shard filename(s) to generate (repeatable)")
    ap.add_argument("--limit-shards", type=int, default=None, help="Only generate the first K shards (for testing)")
    ap.add_argument("--verify-only", action="store_true", help="Only verify existing shard sizes without generating")
    ap.add_argument("--no-sandbox", action="store_true", help="Run snapshot R directly without bwrap")
    ap.add_argument("--system-r", action="store_true", help="Use host system Rscript instead of snapshot runtime")
    a = ap.parse_args(argv)
    res = generate_shards(
        Path(a.task_dir),
        split=a.split,
        cores=a.cores,
        shard_names=a.shards,
        limit_shards=a.limit_shards,
        verify_only=a.verify_only,
        use_system_r=a.system_r,
        no_sandbox=a.no_sandbox,
    )
    if not res["ok"]:
        print(f"Verification failed: {res['missing_or_mismatch'][:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Training script for consistency bridges

Usage:
    python train_bridge.py --config configs/ou.yaml --gpu 0
    python train_bridge.py --config configs/ou.yaml --gpu 0 --train.lr 3e-4 --train.num_outer_iterations 5000
"""

import os
import sys
import argparse
import yaml
import json
import pickle
from datetime import datetime


# helpers

def apply_cli_overrides(cfg, overrides):
    """
    Modifies cfg with overrides
    """
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        d[keys[-1]] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Train a consistency bridge")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs/<problem_name>/<timestamp>_run)")
    # Collect any extra --section.key value pairs as overrides
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i].lstrip("-")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    return args, overrides


def _build_model(model_cfg, d):
    """Build the neural network from the model config."""
    import flax.linen as nn
    from src.models import ScoreMLP, ConservativeMLP

    activation_map = {"relu": nn.relu, "gelu": nn.gelu, "silu": nn.silu, "tanh": nn.tanh}
    activation = activation_map[model_cfg.get("activation", "gelu")]
    dim_hidden = tuple(model_cfg["dim_hidden"])
    emb_dim_hidden = tuple(model_cfg["emb_dim_hidden"])

    model_type = model_cfg.get("type", "conservative")
    if model_type == "conservative":
        return ConservativeMLP(
            dim_hidden=dim_hidden,
            emb_dim_hidden=emb_dim_hidden,
            activation=activation,
        )
    elif model_type == "score":
        return ScoreMLP(
            dim_hidden=dim_hidden,
            emb_dim_hidden=emb_dim_hidden,
            activation=activation,
            out_dim=d,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


###################################
# Single training run
###################################

def run_training(cfg, outdir):
    """Run a single training, evaluate KL metrics, and save artefacts under `outdir`.

    Returns a dict of final scalar metrics.
    """
    import jax
    import jax.numpy as jnp

    from src.consistency_bridge import ConsistencyBridge
    from problems import PROBLEMS

    os.makedirs(outdir, exist_ok=True)

    # save config
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("=" * 60)
    print(f"Output directory: {outdir}")

    # build problem
    problem_name = cfg["problem"]["name"]
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem_name}'. Available: {list(PROBLEMS.keys())}")
    problem = PROBLEMS[problem_name](cfg["problem"])

    # build model
    d = problem["shape"][0]
    model = _build_model(cfg["model"], d)

    # build consistency bridge
    bridge_config = cfg["bridge"]
    bridge = ConsistencyBridge(
        shape=problem["shape"],
        x_0=problem["x_0"],
        x_T=problem["x_T"],
        base_drift_fn=problem["base_drift"],
        sigma_fn=problem["sigma_fn"],
        model=model,
        bridge_config=bridge_config,
        T=problem["T"],
    )

    # train
    train_config = dict(cfg["train"])  # copy so cfg stays intact for saving
    seed = train_config.pop("seed", 0)
    key = jax.random.PRNGKey(seed)
    state, ema_params_lst, ema_grad_norms = bridge.train(key, train_config)

    # save checkpoint
    ckpt_path = os.path.join(outdir, "checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "state": state,
            "ema_params": ema_params_lst[-1],
            "ema_params_lst": ema_params_lst,
        }, f)
    print(f"Saved checkpoint to {ckpt_path}")

    # Target metric is recomputed from this checkpoint in the evaluation stage.
    return {}


###################################
# Main
###################################

def main():
    args, overrides = parse_args()

    # use a particular GPU — set before JAX imports
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # apply any CLI overrides
    apply_cli_overrides(cfg, overrides)

    # set output directory
    problem_name = cfg["problem"]["name"]
    if args.outdir is not None:
        outdir = args.outdir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("outputs", problem_name, f"{timestamp}_run")

    run_training(cfg, outdir)


if __name__ == "__main__":
    main()

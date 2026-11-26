"""Run the APPFL gRPC FedQueue server for the controlled MNIST experiment.

This is a thin wrapper around the official ``ServerAgent`` +
``GRPCServerCommunicator`` (see ``code/examples/grpc/run_server.py``) that adds
a monitor thread which snapshots the aggregated global model after every
FedQueue aggregation round, together with its wall-clock timestamp.  The
snapshots let us compute max accuracy and time-to-target accuracy offline.
"""

import argparse
import json
import os
import threading
import time

import torch
from omegaconf import OmegaConf

from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    config = OmegaConf.load(args.config)
    run_name = os.path.splitext(os.path.basename(args.config))[0]
    results_dir = args.results_dir or os.path.join("results", run_name)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "round_history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    queue_path = os.path.join(results_dir, "queue_draws.jsonl")
    if os.path.exists(queue_path):
        os.remove(queue_path)
    server_agent = ServerAgent(server_agent_config=config)
    scheduler = server_agent.scheduler
    # Observe the unchanged queue RNG for the protocol certificate.
    original_sample = scheduler._sample_queue_delay
    def sample_queue(client_id):
        delay = original_sample(client_id)
        with open(os.path.join(results_dir, "queue_draws.jsonl"), "a") as f:
            f.write(json.dumps({"cid": client_id, "delay": delay}) + "\n")
        return delay
    scheduler._sample_queue_delay = sample_queue
    target_rounds = int(config.server_configs.num_global_epochs)

    t0 = time.time()
    # Called synchronously under the scheduler lock: no missing or mixed rounds.
    def snapshot(round_idx, params):
        state_dict = {k: v.detach().cpu().clone() for k, v in params.items()}
        timestamp = round(time.time() - t0, 3)
        torch.save(state_dict, os.path.join(ckpt_dir, f"round_{round_idx:03d}.pt"))
        with open(history_path, "a") as f:
            f.write(json.dumps({"round": round_idx, "wall_time_s": timestamp}) + "\n")
    scheduler.checkpoint_callback = snapshot

    communicator = GRPCServerCommunicator(
        server_agent,
        logger=server_agent.logger,
        **config.server_configs.comm_configs.grpc_configs,
    )
    serve(communicator, **config.server_configs.comm_configs.grpc_configs)

    with open(os.path.join(results_dir, "run_summary.json"), "w") as f:
        json.dump(
            {
                "aggregation_rounds": scheduler.get_num_global_epochs(),
                "target_rounds": target_rounds,
                "total_wall_time_s": round(time.time() - t0, 3),
            },
            f,
            indent=2,
        )
    print(
        f"[server] finished: {scheduler.get_num_global_epochs()} aggregation rounds "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

"""Run one APPFL gRPC FedQueue client (official ClientAgent loop)."""

import argparse
import os
import random
import numpy as np

import torch
from omegaconf import OmegaConf

from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    client_agent_config = OmegaConf.load(args.config)
    seed = int(os.environ["BENCH_SEED"]) + int(client_agent_config.data_configs.dataset_kwargs.client_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = GRPCClientCommunicator(
        client_id=client_agent.get_id(),
        logger=client_agent.logger,
        **client_agent_config.comm_configs.grpc_configs,
    )

    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    if isinstance(init_global_model, tuple):
        init_global_model, metadata = init_global_model[0], init_global_model[1]
    else:
        metadata = {}
    client_agent.load_parameters(init_global_model)

    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(action="set_sample_size", sample_size=sample_size)

    while True:
        client_agent.train(**metadata)
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")
    print(f"[{client_agent.get_id()}] finished")


if __name__ == "__main__":
    main()

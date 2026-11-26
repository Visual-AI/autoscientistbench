"""Model construction in the same order as the original Experiment 4.1 driver."""
import json
from pathlib import Path
import torch
from model.embeddings import GammaEmbedding, ObservationEmbedding, ComponentEmbedding
from model.distribution_transformer import DistributionTransformer


def build_model(config=None):
    if config is None:
        raise ValueError('config is required')
    d = config['transformer_kwargs']['d_model']
    prior = GammaEmbedding(d_model=d, n_components=config['n_components'],
                           **config['distribution_embedding_kwargs'])
    obs = {k: ObservationEmbedding(d_model=d, observation_size=1, **v)
           for k, v in config['observation_embedding_kwargs'].items()}
    component = ComponentEmbedding(state_size=1, d_model=d, **config['component_embedding_kwargs'])
    return DistributionTransformer(component_embedding=component,
        transformer_kwargs=config['transformer_kwargs'], n_components=config['n_components'],
        prior_embedding=prior, sample_space_transform=torch.log, **obs)


def load_model(model_dir):
    config = json.loads((Path(model_dir) / 'config.json').read_text())
    model = build_model(config)
    model.load_state_dict(torch.load(Path(model_dir) / 'model.pt', map_location='cpu', weights_only=True))
    model.train()  # Original checkpoint evaluation leaves training=True; dropout is zero.
    return model


def forward(model, phi, z):
    return model(phi, obs_1=z)[1]

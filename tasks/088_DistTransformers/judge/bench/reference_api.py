"""Frozen original architecture, used for RNG alignment and same-job timing."""
import json
from pathlib import Path
import torch
from _dt_reference.model.embeddings import GammaEmbedding, ObservationEmbedding, ComponentEmbedding
from _dt_reference.model.distribution_transformer import DistributionTransformer


def build_model(config=None):
    if config is None:
        config = json.loads((Path(__file__).parent / 'config.json').read_text())
    d = config['transformer_kwargs']['d_model']
    prior = GammaEmbedding(d_model=d, n_components=config['n_components'],
                           **config['distribution_embedding_kwargs'])
    obs = {k: ObservationEmbedding(d_model=d, observation_size=1, **v)
           for k, v in config['observation_embedding_kwargs'].items()}
    component = ComponentEmbedding(state_size=1, d_model=d, **config['component_embedding_kwargs'])
    return DistributionTransformer(component_embedding=component,
        transformer_kwargs=config['transformer_kwargs'], n_components=config['n_components'],
        prior_embedding=prior, sample_space_transform=torch.log, **obs)

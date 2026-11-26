"""Original reusable training; external data entry and final checkpoint export."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from model_api import build_model
from distributions.distributions import InverseGammaMetaPrior, MappedScaleGaussianObservationModel, CompleteDistribution
from workflows.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get('BENCH_SEED', '1'))
    cpus = int(os.environ.get('BENCH_CPUS', '8'))
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    config = json.loads((Path(os.environ['BENCH_DATA_DIR'])/'train/config.json').read_text())
    meta = InverseGammaMetaPrior(**config['meta_prior_kwargs'])
    obs = {k: MappedScaleGaussianObservationModel(torch.tensor(v, dtype=torch.float32),
               scale_parametrisation='covariance_matrix', mapping=None)
           for k,v in config['observation_loc'].items()}
    distribution = CompleteDistribution(meta, **obs)
    model = build_model(config)
    model, metrics = train(model, distribution, **config['training_kwargs'])
    torch.save(model.state_dict(), a.out/'model.pt')
    (a.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (a.out/'certificate.json').write_text(json.dumps(dict(seed=seed,
        n_components=model.n_components, amortised=True,
        parameters=sum(p.numel() for p in model.parameters()),
        epochs=config['training_kwargs']['epochs']),indent=2)+'\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Operator probe: bind predictions to a saved fitted-state artifact.
Legacy tensor manifests are checked when supplied. Artifact completeness and
complete learned-parameter declarations and actual training-work accounting require patch audit.
Usage: check_model.py MODEL_DIR --pred TABLE [--report JSON]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prediction import read_predictions
from scorer import PARAMETER_LIMIT


def check(model, pred):
    card = json.loads((model/'model_card.json').read_text())
    assert card['format'] in ('hsg-model-v1', 'method-model-v1'), 'wrong certificate format'
    artifact = model/card.get('artifact_file', 'weights.pt')
    assert artifact.resolve().is_relative_to(model.resolve()), 'artifact outside model directory'
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert card['weights_sha256'] == digest, 'artifact digest mismatch'
    count = card['trainable_parameters']
    assert all(isinstance(card[k], int) and not isinstance(card[k], bool) and card[k]>=0
               for k in ('trainable_parameters','optimizer_steps','epochs_started')), 'invalid diagnostic counts'
    if card['format'] == 'hsg-model-v1':
        state = torch.load(artifact, map_location='cpu', weights_only=True)
        names = card['parameters']; buffers = card.get('buffers', {})
        assert isinstance(names, dict) and isinstance(buffers, dict), 'invalid state manifests'
        assert not (set(names)&set(buffers)) and set(state)==set(names)|set(buffers), 'unaccounted tensors'
        actual=0
        for name, shape in {**names,**buffers}.items():
            assert list(state[name].shape)==shape and torch.isfinite(state[name]).all(), 'invalid state tensor'
            if name in names: actual += state[name].numel() * (2 if state[name].is_complex() else 1)
        assert actual == count, 'parameter diagnostic mismatch'
    else:
        parameter_file=model/card.get('parameter_artifact_file','trainable_parameters.npz')
        assert parameter_file.resolve().is_relative_to(model.resolve()), 'parameters outside model directory'
        actual=0
        with np.load(parameter_file,allow_pickle=False) as archive:
            for name in archive.files:
                value=archive[name]
                assert np.issubdtype(value.dtype,np.number) and np.isfinite(value).all(), 'invalid learned parameter array'
                actual += value.size * (2 if np.iscomplexobj(value) else 1)
        assert actual == count, 'learned-parameter manifest mismatch'
    assert count <= PARAMETER_LIMIT, 'capacity budget exceeded'
    assert card['optimizer_steps'] <= 1000 and card['epochs_started'] <= 100, 'training-work budget exceeded'

    _, meta = read_predictions(pred)
    assert meta == (count, card['optimizer_steps'], card['epochs_started'], digest), 'prediction certificate mismatch'
    return {'pass':True,'parameters':count,'steps':card['optimizer_steps'],'epochs':card['epochs_started']}

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model',type=Path);ap.add_argument('--pred',required=True);ap.add_argument('--input',type=Path);ap.add_argument('--report',type=Path)
    a=ap.parse_args()
    try:
        if a.input:
            from validate import main as validate
            import contextlib
            with contextlib.redirect_stdout(sys.stderr): validate(a.pred, a.input)
        result=check(a.model,a.pred)
    except Exception as e: result={'pass':False,'accuracy':float('inf'),'reason':str(e)}
    print(json.dumps(result))
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    sys.exit(0 if result['pass'] else 1)

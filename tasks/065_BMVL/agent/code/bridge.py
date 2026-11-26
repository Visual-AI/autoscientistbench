"""BMVL baseline over a single authorized fit or inference request.

All real fitting happens inside private evaluation. This module deliberately has
no loader for the complete research datasets, split generation, or test labels.
The original model, training corruption, loss, optimizer and schedule are reused.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def fit(request, root, model_dir, output):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from multi_view import MultiViewBackbone, NC_MultiViewDataset, set_seed
    config = json.loads((model_dir / 'config.json').read_text())
    with np.load(root / 'train.npz', allow_pickle=False) as arrays:
        views = [arrays[f'x{i}'] for i in range(request['num_views'])]
        labels = arrays['y']
    args = SimpleNamespace(**config, dims_list=[x.shape[1] for x in views],
                           num_classes=request['num_classes'])
    set_seed(request['fit_seed'])
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS', '1')))
    model = MultiViewBackbone(args).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    ce = torch.nn.CrossEntropyLoss()
    indices = np.arange(len(labels))
    for epoch in range(args.epochs):
        model.train()
        data = NC_MultiViewDataset(views, labels, indices, noise_seed=epoch, noise_ratio=args.augment_ratio)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        for view_data, y, indicators in loader:
            _, logits, reliability = model(view_data)
            loss = ce(logits, y) + args.lambda_w * F.binary_cross_entropy(reliability, indicators.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
    torch.save({'state': model.state_dict(), 'args': vars(args)}, output)


def infer(request, root, output):
    import numpy as np
    import torch
    from multi_view import MultiViewBackbone
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS', '1')))
    # Candidate fit output is interpreted only inside the isolated candidate.
    saved = torch.load(root / 'fit.out', map_location='cpu')
    args = SimpleNamespace(**saved['args'])
    model = MultiViewBackbone(args).cpu()
    model.load_state_dict(saved['state'])
    model.eval()
    with np.load(root / 'input.npz', allow_pickle=False) as arrays:
        views = [arrays[f'x{i}'] for i in range(request['num_views'])]
    batches = []
    with torch.no_grad():
        for start in range(0, len(views[0]), args.batch_size):
            xs = [torch.from_numpy(x[start:start+args.batch_size]) for x in views]
            _, logits, _ = model(xs)
            batches.extend(torch.argmax(logits, dim=1).tolist())
    output.write_text(json.dumps({'pred': batches}, separators=(',', ':')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'export', 'policy'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'train':
        args.out.mkdir(parents=True, exist_ok=True)
        config = json.loads(Path('config.json').read_text())
        (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    elif args.stage == 'export':
        from private_eval import export_artifact
        request = json.loads(args.input.read_text())
        if request.get('task_id') != 'cvpr-427-BMVL': raise ValueError('wrong task')
        code=Path(__file__).parent
        reference=Path(os.environ.get('BENCH_BENCH_DIR',code.parent/'bench'))/'reference_env.lock'
        # The trusted scorer runs in the frozen reference environment. Reuse its
        # packages when pins are identical; changed candidate pins remain fully
        # exported by the canonical runtime so dependency freedom is preserved.
        unchanged=reference.is_file() and (code/'env.lock').read_bytes()==reference.read_bytes()
        export_artifact(code, args.model, args.output, 'cvpr-427-BMVL',include_dependencies=not unchanged)
    else:
        request = json.loads((args.input/'request.json').read_text())
        if request['stage'] == 'fit': fit(request, args.input, args.model, args.output)
        elif request['stage'] == 'predict': infer(request, args.input, args.output)
        else: raise ValueError('unknown stage')

if __name__ == '__main__': main()

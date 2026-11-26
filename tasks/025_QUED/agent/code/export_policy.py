"""Export inert source and training recipe for isolated fresh fitting."""
import argparse
from pathlib import Path
from private_eval import export_artifact

p = argparse.ArgumentParser()
for name in ('model', 'input', 'output'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
export_artifact(Path(__file__).resolve().parent, a.model, a.output, 'Che-06', include_dependencies=False)

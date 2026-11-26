"""Write the baseline configuration: RPB-OM with POPU at the paper's settings (Table 1). Nothing is fitted."""
import argparse
import json
from pathlib import Path

CONFIG = {'algorithm': 'RPB-OM', 'predictor': 'POPU', 'tau': 4, 'max_support_factor': 100}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out, 'config.json').write_text(json.dumps(CONFIG) + '\n')

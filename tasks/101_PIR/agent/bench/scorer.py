"""Protocol-v2 scoring: isolated inference, then the original numeric metric."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_eval import main

if __name__ == '__main__':
    main()

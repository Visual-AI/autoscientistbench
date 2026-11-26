#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_json,check_prediction,metric
if __name__=="__main__":
 directory=Path(sys.argv[2]);here=Path(__file__).resolve().parent
 instances=read_json(directory/"manifest.json")["instances"]
 predictions=check_prediction(read_json(sys.argv[1]),instances)
 # Compute fixed original Lasso selection independently of candidate outputs.
 process=subprocess.run(["Rscript",str(here/"oracle.R"),str(directory)],check=True,text=True,capture_output=True)
 oracle={row["id"]:row["FDP"] for row in json.loads(process.stdout)}
 baseline={row["design"]:row["baseline_mean_sd"] for row in read_json(here/"calibration.json")}
 print(json.dumps(metric(predictions,oracle,instances,baseline),allow_nan=True))

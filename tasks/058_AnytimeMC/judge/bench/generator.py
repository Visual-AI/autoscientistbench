#!/usr/bin/env python3
"""Offline original Gaussian DGP; operator supplies a split seed. Never runs a method."""
import argparse,csv,hashlib,json,subprocess,uuid
from pathlib import Path

def generate(out,seed,scope):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    M,B,n,cell=(1000,10000,10,4) if scope=="full" else (24,256,2,1)
    ids=[uuid.uuid4().hex for _ in range(n)]
    (out/"ids.txt").write_text("\n".join(ids)+"\n")
    subprocess.run(["Rscript","--vanilla",str(Path(__file__).with_suffix(".R")),str(out),str(M),str(B),str(n),str(seed),str(cell),str(out/"ids.txt")],check=True)
    truth={i:[] for i in ids}
    with (out/"truth.csv").open() as f:
        for row in csv.DictReader(f): truth[row["trial_id"]].append(int(row["alternative"]))
    labels={"version":1,"scope":scope,"protocol":{"M":M,"B":B,"trials":n,"pi_A":0.4,"mu_A":2.5,"alpha":0.1,"h":10,"b":0.9},
      "input_sha256":hashlib.sha256((out/"input.csv").read_bytes()).hexdigest(),
      "trials":[{"trial_id":i,"alternative":truth[i]} for i in ids]}
    (out/"labels.json").write_text(json.dumps(labels,indent=2)+"\n")
    (out/"truth.csv").unlink();(out/"ids.txt").unlink()
    print(json.dumps({"scope":scope,"M":M,"B":B,"trials":n,"input_sha256":labels["input_sha256"]}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--scope",choices=["full","smoke"],default="full")
    a=p.parse_args();generate(a.out,a.seed,a.scope)

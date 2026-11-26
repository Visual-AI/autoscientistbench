"""Frozen, stdlib-only prediction parser shared by validation and scoring."""
import csv,math,re
ID=re.compile(r"[0-9a-f]{32}\Z")
def predictions(path):
    values={}
    with open(path,newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ["id","alpha_hat_adj"]:raise ValueError("expected id,alpha_hat_adj columns")
        for row in reader:
            if set(row)!={"id","alpha_hat_adj"} or not ID.fullmatch(row["id"]):raise ValueError("invalid row or ID")
            value=float(row["alpha_hat_adj"])
            if not math.isfinite(value) or not 0<=value<=math.pi:raise ValueError("angle outside finite [0,pi]")
            if row["id"] in values:raise ValueError("duplicate prediction ID")
            values[row["id"]]=value
    if not values:raise ValueError("empty predictions")
    return values

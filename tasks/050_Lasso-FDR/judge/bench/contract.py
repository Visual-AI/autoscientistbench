"""Frozen prediction contract and metric guard arithmetic; Python standard library."""
import json,math,statistics
DESIGNS=("IID_Normal","X_AR","Coef_AR","Sparse")
def unique_object(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise ValueError("duplicate JSON key: "+key)
  out[key]=value
 return out
def read_json(path):
 with open(path) as f:return json.load(f,object_pairs_hook=unique_object)
def numeric(x):
 return type(x) in (int,float) and math.isfinite(x)
def check_prediction(pred,instances):
 if type(pred) is not dict or pred.get("version")!=1 or type(pred.get("estimates")) is not list:
  raise ValueError("expected version=1 and estimates list")
 expected={r["id"]:r for r in instances}
 if len(expected)!=len(instances):raise ValueError("duplicate frozen instance IDs")
 seen={}
 for row in pred["estimates"]:
  if type(row) is not dict or type(row.get("id")) is not str or row["id"] not in expected or row["id"] in seen:
   raise ValueError("unknown, missing, or duplicate prediction ID")
  values=row.get("hFDR")
  if type(values) is not list or len(values)!=expected[row["id"]]["n_lambda"] or any(not numeric(x) or x<0 for x in values):
   raise ValueError("hFDR must contain one finite nonnegative number per lambda")
  seen[row["id"]]=values
 if set(seen)!=set(expected):raise ValueError("prediction ID set does not match observations")
 return seen
def metric(predictions,oracle,instances,baseline):
 groups={g:[] for g in DESIGNS}
 for item in instances:groups[item["design"]].append(item)
 failures=[];cells=[];design_sd={}
 for design,items in groups.items():
  if len(items)!=120 or any(item["n_lambda"]!=10 for item in items):
   failures.append("full_protocol:"+design)
  if not items:continue
  nlam=items[0]["n_lambda"]
  sds=[]
  for k in range(nlam):
   h=[predictions[item["id"]][k] for item in items]
   f=[oracle[item["id"]][k] for item in items]
   if len(h)<2:
    failures.append("insufficient_replicates:"+design);continue
   differences=[a-b for a,b in zip(h,f)]
   bias=statistics.mean(differences)
   se=statistics.stdev(differences)/math.sqrt(len(h))
   sd=statistics.stdev(h);sds.append(sd)
   if bias < -3*se:failures.append("bias:"+design+":"+str(k+1))
   cells.append({"design":design,"lambda_index":k+1,"bias":bias,"se_cell":se,"sd_hFDR":sd})
  if sds:
   design_sd[design]=statistics.mean(sds)
   if design_sd[design] > 1.2*baseline[design]:failures.append("sd:"+design)
 mad=statistics.mean(abs(cell["bias"]) for cell in cells) if cells else None
 return {"MAD":float("inf") if failures or len(cells)!=40 else mad,"unpenalized_MAD":mad,"guard_failures":failures,"mean_sd_by_design":design_sd,"cells":cells}

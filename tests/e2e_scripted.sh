#!/usr/bin/env bash
# End-to-end on the toy task with the scripted harness (no model calls):
#   fixture -> lint -> reference -> closure probe run -> improving run -> judge (expect PASS) -> empty patch (expect no PASS)
set -euo pipefail
cd "$(dirname "$0")/.."
export ARB_LOCAL_ROOT="${ARB_LOCAL_ROOT:-/tmp/arb}"
CPUS="${CPUS:-2}"
STAMP="$(date +%H%M%S)"
B="python3 -m bench"

echo "### fixture + lint + reference"
$B fixture toy-ridge --force
$B lint toy-ridge
$B reference toy-ridge --cpus "$CPUS"
$B lint toy-ridge --full

echo "### closure probe"
$B run --task toy-ridge --harness scripted --set script=harnesses/scripted/probe.sh --budget-hours 0.05 --cpus "$CPUS" \
   --suite e2e --run-id "e2e-probe-$STAMP" || [ $? = 2 ]   # the probe submits nothing: bench run exits 2
echo "--- probe output"; cat "runs/e2e-probe-$STAMP/agent_events.jsonl"
grep -q 'ls: cannot access\|没有那个文件' "runs/e2e-probe-$STAMP/agent_events.jsonl" || { echo "FAIL: /mnt visible?"; exit 1; }
grep -q '^connect=403' "runs/e2e-probe-$STAMP/agent_events.jsonl" || { echo "FAIL: example.com not blocked"; exit 1; }
grep -q '^connect=200 http=200' "runs/e2e-probe-$STAMP/agent_events.jsonl" || { echo "FAIL: pypi not reachable via proxy"; exit 1; }

echo "### improving scripted agent"
$B run --task toy-ridge --harness scripted --budget-hours 0.1 --cpus "$CPUS" --suite e2e --run-id "e2e-improve-$STAMP"
echo "--- agent output"; tail -n 30 "runs/e2e-improve-$STAMP/agent_events.jsonl"
$B judge --run "e2e-improve-$STAMP" --cpus "$CPUS"
python3 - "runs/e2e-improve-$STAMP/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("status", r["status"], "gain", r["gain"], "threshold", r["threshold"], "pass", r["pass"])
assert r["status"] == "scored", r.get("failure")
assert r["pass"] is True, "expected PASS for the poly+sin patch"
assert r["reference"]["parity"].get("ok", True), r["reference"]["parity"]
PY

echo "### empty patch must not pass (gain ~ 0)"
: > "$ARB_LOCAL_ROOT/empty.patch"
$B judge --task toy-ridge --patch "$ARB_LOCAL_ROOT/empty.patch" --cpus "$CPUS" --label "empty-$STAMP" || true
python3 - "runs/judge__toy-ridge__empty-$STAMP/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("status", r["status"], "gain", r["gain"], "pass", r["pass"])
assert r["status"] == "scored" and abs(r["gain"]) < 1e-9 and r["pass"] is False
PY
echo "### e2e OK"

"""Frozen process-tree resource monitor and prediction formatter.

The independent resource probe must still verify that submitted launchers call
this monitor around all computation. A certificate alone is not trusted proof.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def tree_peak_rss(root):
    records = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields = {}
            for line in (p / 'status').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in {'PPid', 'VmHWM'}: fields[key] = int(value.split()[0])
            records[int(p.name)] = (fields.get('PPid', -1), fields.get('VmHWM', 0) * 1024)
        except (OSError, ValueError, IndexError): pass
    children = {root}
    while True:
        more = {pid for pid, (parent, _) in records.items() if parent in children}
        new = children | more
        if new == children: break
        children = new
    return sum(records.get(pid, (0, 0))[1] for pid in children)


def monitor(command, report):
    start = time.monotonic(); peak = 0
    child = subprocess.Popen(command)
    while child.poll() is None:
        peak = max(peak, tree_peak_rss(int(Path("/proc/self/status").read_text().split("\nPid:")[1].split()[0])))
        time.sleep(.1)
    elapsed = time.monotonic() - start
    peak = max(peak, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
    obj = {'wall_s': elapsed, 'peak_gb': peak / 1e9, 'cpus': int(os.environ.get('BENCH_CPUS', '64')),
           'measurement': 'sum of live process VmHWM; 0.1 s sampling; GB=1e9 bytes', 'returncode': child.returncode}
    Path(report).write_text(json.dumps(obj, indent=2)+'\n')
    return child.returncode


def format_prediction(raw, output, training, prediction):
    t = json.loads(Path(training).read_text()); p = json.loads(Path(prediction).read_text())
    assert t['returncode'] == p['returncode'] == 0
    with open(raw, newline='') as src, open(output, 'w', newline='') as dst:
        reader=csv.DictReader(src); writer=csv.writer(dst)
        writer.writerow(['id','pred','train_s','peak_gb','cpus'])
        for row in reader:
            writer.writerow([row['id'],row['pred'],t['wall_s'],max(t['peak_gb'],p['peak_gb']),t['cpus']])


if __name__ == '__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    m=s.add_parser('monitor');m.add_argument('--report',required=True);m.add_argument('cmd',nargs=argparse.REMAINDER)
    f=s.add_parser('format')
    for name in ['raw','output','training','prediction']:f.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.mode=='monitor':raise SystemExit(monitor(a.cmd[1:] if a.cmd[:1]==['--'] else a.cmd,a.report))
    format_prediction(a.raw,a.output,a.training,a.prediction)

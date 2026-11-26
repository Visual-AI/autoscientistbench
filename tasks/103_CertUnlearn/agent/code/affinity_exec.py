"""Set XLA's visible CPU pool before its import, respecting the allocated budget."""
import os,sys
n=int(os.environ['BENCH_CPUS'])
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:n])
os.execv(sys.executable,[sys.executable,*sys.argv[1:]])

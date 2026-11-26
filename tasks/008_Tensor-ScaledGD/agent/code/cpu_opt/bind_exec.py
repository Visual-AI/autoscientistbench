"""Launch Octave, deferring its MKL preload until after Python startup.

--no-bind exercises the same exec path in development without changing affinity.
"""
import ctypes
import os
import sys

if sys.argv[1] == '--no-bind':
    command = sys.argv[2:]
    print('placement disabled for smoke/dev; affinity unchanged', flush=True)
else:
    node, cpus, *command = sys.argv[1:]
    assert int(os.environ.get('REPRO_THREADS', '0')) > 0
    assert os.environ.get('REPRO_SMOKE') != '1'
    selected = {int(x) for x in cpus.split(',')}
    assert selected <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, selected)
    # numactl is not installed. Use its libnuma interface before allocation.
    lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
    mask = ctypes.c_ulong(1 << int(node))
    lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    rc = lib.set_mempolicy(2, ctypes.byref(mask), 64)  # MPOL_BIND
    print(f'placement node={node} cpus={cpus} membind_rc={rc} errno={ctypes.get_errno()}', flush=True)
    if rc:
        print('membind unavailable; pinned first-touch allocation remains node-local', flush=True)
env = dict(os.environ)
preload = env.pop('REPRO_OCTAVE_PRELOAD', '')
if preload:
    env['LD_PRELOAD'] = preload
os.execvpe(command[0], command, env)

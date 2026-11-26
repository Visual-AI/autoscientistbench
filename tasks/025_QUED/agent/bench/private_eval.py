"""Task-owned policy export and private execution.

Artifacts are inert files until executed in a fresh filesystem/network sandbox.
The private scorer never imports candidate code or unpickles candidate models.
"""
import argparse, contextlib, ctypes, json, math, os, re, shutil, signal, socket, stat
import subprocess, sys, sysconfig, tempfile, threading, time, zipfile
from pathlib import Path, PurePosixPath
from private_wire import recv_frame, send_frame
PROTOCOL = 'private-policy-v2'
MAX_BYTES = 64 * 1024 ** 3
MAX_FILES = 400000
MAX_OUTPUT = 2 * 1024 ** 3
HERE = Path(__file__).resolve().parent
SYSTEM = ('/usr', '/lib', '/lib64', '/lib32', '/bin', '/sbin')

def remaining_time():
    """Read the trusted evaluation deadline; standalone operator probes may omit it."""
    value = os.environ.get('BENCH_EVAL_DEADLINE_TS')
    if value is None: return None
    deadline = float(value)
    if not math.isfinite(deadline): raise ValueError('invalid evaluation deadline')
    remaining = deadline - time.time()
    if remaining <= 0: raise TimeoutError('overall evaluation budget exhausted')
    return remaining

def _valid_member(name):
    p = PurePosixPath(name)
    if '\\' in name or '\x00' in name or p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('unsafe archive path')
    if str(p) != name.rstrip('/') or p.parts[0] not in ('code','model','resources','deps','manifest.json'):
        raise ValueError('unexpected archive member')
    return p

def _tree_files(root):
    root = Path(root)
    if not root.is_dir(): raise ValueError('artifact source missing: ' + str(root))
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(n for n in names if n not in ('.git','__pycache__','.pytest_cache','.mypy_cache','.venv'))
        for name in sorted(files):
            if name.endswith(('.pyc','.pyo')): continue
            path = Path(directory) / name
            if path.is_file(): yield path, path.relative_to(root).as_posix()

def _dependency_files(code_dir):
    """Copy locked distributions from the running candidate interpreter only."""
    from importlib import metadata
    lock = Path(code_dir)/'env.lock'
    pins = []
    if lock.is_file():
        pins = [m.group(1) for m in re.finditer(r'^\s*([A-Za-z0-9_.-]+)==',lock.read_text(),re.M)]
    if not pins:
        for path,relative in _tree_files(sysconfig.get_paths()['purelib']):
            yield path,'deps/site-packages/'+relative
        return
    seen = set()
    for name in pins:
        dist = metadata.distribution(name)
        for item in dist.files or ():
            rel = PurePosixPath(str(item))
            if rel.is_absolute() or '..' in rel.parts or str(rel).endswith(('.pyc','.pyo')): continue
            path = Path(dist.locate_file(item))
            target = 'deps/site-packages/'+str(rel)
            if path.is_file() and target not in seen:
                seen.add(target); yield path,target

def export_artifact(code_dir, model_dir, output, task_id, resources=None, include_dependencies=True):
    code_dir, model_dir, output = Path(code_dir), Path(model_dir), Path(output)
    if not (code_dir / 'policy.sh').is_file(): raise ValueError('code/policy.sh required')
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [('code',code_dir), ('model',model_dir)]
    if resources:
        if isinstance(resources, dict):
            for key,value in resources.items():
                if not key or '/' in key or key in ('.','..'): raise ValueError('invalid resource name')
                sources.append(('resources/' + key, Path(value)))
        else: sources.append(('resources',Path(resources)))
    manifest = {'protocol':PROTOCOL, 'task_id':str(task_id), 'python':list(sys.version_info[:2]),
                'algorithm_seed':int(os.environ.get('BENCH_SEED','0')), 'dependencies_included':bool(include_dependencies)}
    total = count = 0
    names = {'manifest.json'}
    partial = output.with_name(output.name + '.partial-' + str(os.getpid()))
    excluded = {os.path.abspath(output), os.path.abspath(partial)}
    try:
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True))
            file_groups = ([(path,prefix+'/'+relative) for path,relative in _tree_files(source)] for prefix,source in sources)
            if include_dependencies:
                import itertools
                file_groups = itertools.chain(file_groups,[_dependency_files(code_dir)])
            for group in file_groups:
                for path,name in group:
                    if os.path.abspath(path) in excluded: continue
                    if name in names: continue
                    _valid_member(name)
                    file_stat = path.stat()
                    total += file_stat.st_size; count += 1
                    if total > MAX_BYTES or count > MAX_FILES: raise ValueError('artifact too large')
                    info = zipfile.ZipInfo(name)
                    info.external_attr = (stat.S_IFREG | (0o755 if file_stat.st_mode & 0o111 else 0o644)) << 16
                    with archive.open(info,'w',force_zip64=True) as dest, path.open('rb') as src:
                        shutil.copyfileobj(src,dest,1024*1024)
                    names.add(name)
        os.replace(partial,output)
    finally: partial.unlink(missing_ok=True)
    return manifest

def validate_artifact(path, task_id):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES + 1024**3:
        raise ValueError('invalid artifact file')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not 2 <= len(members) <= MAX_FILES: raise ValueError('invalid member count')
        seen = set(); total = 0
        for member in members:
            _valid_member(member.filename)
            if member.filename in seen: raise ValueError('duplicate member')
            seen.add(member.filename)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if member.is_dir() or kind not in (0,stat.S_IFREG): raise ValueError('regular files required')
            if member.flag_bits & 1 or member.compress_type != zipfile.ZIP_STORED:
                raise ValueError('unencrypted stored archive required')
            total += member.file_size
            if total > MAX_BYTES or member.file_size < 0: raise ValueError('artifact too large')
        if 'manifest.json' not in seen or 'code/policy.sh' not in seen: raise ValueError('entry point missing')
        if archive.getinfo('manifest.json').file_size > 16384: raise ValueError('manifest too large')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('protocol') != PROTOCOL or manifest.get('task_id') != task_id:
            raise ValueError('task/protocol mismatch')
        if not isinstance(manifest.get('algorithm_seed'),int): raise ValueError('missing algorithm seed')
    return manifest

def _extract(path,target):
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            dest = target / member.filename
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as src, dest.open('xb') as dst: shutil.copyfileobj(src,dst,1024*1024)
            dest.chmod(0o755 if (member.external_attr >> 16) & 0o111 else 0o644)
    for name in ('code','model','resources','deps'): (target/name).mkdir(exist_ok=True)

def _readonly_system():
    flags = []
    for path in SYSTEM:
        if os.path.islink(path): flags += ['--symlink',os.readlink(path),path]
        elif os.path.isdir(path): flags += ['--ro-bind',path,path]
    return flags

def _safe_output(source,destination):
    fd = os.open(source,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as src:
        info = os.fstat(src.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OUTPUT: raise ValueError('invalid policy output')
        with destination.open('xb') as dst:
            left = MAX_OUTPUT + 1
            while left:
                chunk = src.read(min(left,1024*1024))
                if not chunk: break
                dst.write(chunk); left -= len(chunk)
            if not left: raise ValueError('output too large')

class _Broker:
    def __init__(self,path,handler):
        self.path,self.handler = path,handler
        self.stop = threading.Event(); self.error = None; self.calls = 0; self.connection = None
        self.server = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.server.bind(str(path)); self.server.listen(1); self.server.settimeout(0.2)
        self.thread = threading.Thread(target=self._serve,daemon=True); self.thread.start()

    def _serve(self):
        try:
            while not self.stop.is_set():
                try:
                    connection,_ = self.server.accept(); break
                except socket.timeout: continue
            else: return
            self.connection = connection
            with connection:
                while not self.stop.is_set():
                    try: request = recv_frame(connection)
                    except EOFError: break
                    if not isinstance(request,dict) or set(request) != {'method','args','kwargs'}:
                        raise ValueError('invalid request')
                    method,args,kwargs = request['method'],request['args'],request['kwargs']
                    if not isinstance(method,str) or len(method)>128 or not isinstance(args,list) or not isinstance(kwargs,dict):
                        raise ValueError('invalid request')
                    self.calls += 1
                    if self.calls > 5000000: raise ValueError('RPC call safety limit')
                    try: result = self.handler(method,args,kwargs)
                    except Exception as exc:
                        self.error = type(exc).__name__
                        send_frame(connection,{'ok':False,'error':self.error}); return
                    send_frame(connection,{'ok':True,'result':result})
        except Exception as exc:
            if not self.stop.is_set(): self.error = type(exc).__name__

    def close(self):
        self.stop.set()
        if self.connection is not None:
            with contextlib.suppress(OSError): self.connection.shutdown(socket.SHUT_RDWR)
        self.server.close(); self.thread.join(timeout=2)

class Evaluator:
    def __init__(self,pred_path,task_id,cpus,timeout_s=None,runtime_mounts=(),env=None,data_dir=None):
        self.task_id,self.cpus = task_id,max(1,int(cpus))
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.started = time.monotonic()
        remaining_time()
        self.root = Path(tempfile.mkdtemp(prefix='pe-')); self.package = self.root/'package'; self.package.mkdir()
        # Freeze the untrusted output before validation/extraction: candidate
        # descendants must not race validation with a replacement archive.
        snapshot = self.root/'artifact.zip'
        fd = os.open(pred_path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as source, snapshot.open('xb') as dest:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES+1024**3:
                raise ValueError('invalid artifact source')
            remaining = MAX_BYTES+1024**3+1
            while remaining:
                chunk = source.read(min(remaining,1024*1024))
                if not chunk: break
                dest.write(chunk); remaining -= len(chunk)
            if not remaining: raise ValueError('artifact source too large')
        self.manifest = validate_artifact(snapshot,task_id)
        _extract(snapshot,self.package)
        snapshot.unlink()
        self.runtime_mounts = list(runtime_mounts); self.env = dict(env or {})
        self.data_dir = Path(data_dir).resolve() if data_dir else self.package/'resources'
        settings = HERE/'private_eval_runtime.json'
        self.settings = json.loads(settings.read_text()) if settings.exists() else {}
        self.runtime_mounts += self.settings.get('mounts',[])
        self.env = {**self.settings.get('env',{}),**self.env}
        libc = ctypes.CDLL(None,use_errno=True)
        if libc.prctl(4,0,0,0,0) != 0: raise RuntimeError('cannot protect evaluator memory')

    def __enter__(self): return self
    def __exit__(self,*args): shutil.rmtree(self.root,ignore_errors=True)

    def run(self,input_path,handler=None,env=None,timeout_s=None,affinity=None):
        work = Path(tempfile.mkdtemp(prefix='case-',dir=self.root))
        source_input = Path(input_path).resolve()
        snapshot_input = work/'input'
        if source_input.is_dir(): shutil.copytree(source_input,snapshot_input)
        elif source_input.is_file(): shutil.copy2(source_input,snapshot_input)
        else: raise ValueError('policy input must be a regular file or directory')
        output = work/'output'; output.mkdir()
        private_tmp = work/'tmp'; private_tmp.mkdir()
        broker = _Broker(work/'r.sock',handler) if handler is not None else None
        bwrap = HERE/'private-bwrap'
        if not bwrap.is_file(): raise RuntimeError('task private-bwrap binary missing')
        # Workers share this process's CPUs: the list only sets how many threads a worker gets.
        affinity = sorted(os.sched_getaffinity(0))[:self.cpus] if affinity is None else list(affinity)
        if not affinity or len(affinity) > self.cpus:
            raise ValueError('invalid worker CPU allocation')
        # A task may divide its CPU allocation between independent workers.
        # Advertise only this worker's actual allocation, including when an
        # operator runs a bounded smoke inside a smaller parent affinity.
        worker_cpus = len(set(affinity))
        cmd = [str(bwrap),*_readonly_system(),
               '--dev','/dev','--tmpfs','/etc','--tmpfs','/home/policy','--bind',str(private_tmp),'/tmp',
               '--unshare-net','--unshare-ipc','--unshare-pid','--die-with-parent','--cap-drop','ALL']
        for source in ('/proc/cpuinfo','/proc/meminfo'):
            if Path(source).is_file(): cmd += ['--ro-bind',source,source]
        if Path('/usr/bin/which').is_file():
            cmd += ['--ro-bind',os.path.realpath('/usr/bin/which'),'/etc/alternatives/which']
        binds = [(self.package/'code','/ws/code'),(self.package/'model','/ws/model'),
                 (self.package/'deps','/ws/deps'),(self.data_dir,'/ws/resources'),
                 (HERE,'/ws/bench'),(snapshot_input,'/ws/input')]
        prefixes = {Path(sys.prefix),Path(sys.base_prefix),Path(sys.executable).resolve().parent.parent}
        if Path(sys.prefix).is_dir(): binds.append((Path(sys.prefix),'/venv'))
        for prefix in prefixes:
            if str(prefix) != '/' and not str(prefix).startswith(('/usr/','/lib/','/bin/')) and str(prefix) not in SYSTEM:
                binds.append((prefix,str(prefix)))
        for mount in self.runtime_mounts:
            source,target = (mount,mount) if isinstance(mount,str) else mount
            if not Path(source).exists(): raise RuntimeError('runtime mount missing: '+str(source))
            binds.append((Path(source),str(target)))
        done = set()
        for source,target in binds:
            pair = (str(source),str(target))
            if pair not in done: cmd += ['--ro-bind',*pair]; done.add(pair)
        cmd += ['--bind',str(output),'/ws/output']
        if broker: cmd += ['--ro-bind',str(broker.path),'/run/private-eval/rpc.sock']
        inherited = {k:os.environ[k] for k in ('LD_LIBRARY_PATH','R_HOME','R_LIBS_USER','R_LIBS','OCTAVE_HOME') if k in os.environ}
        extra = {**inherited,**self.env,**(env or {})}
        runtime_path = extra.get('PATH',os.environ.get('PATH','/usr/local/bin:/usr/bin:/bin'))
        fixed = {'PATH':'/venv/bin:/ws/deps/bin:'+runtime_path+':/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
                 'HOME':'/home/policy','TMPDIR':'/tmp','TMP':'/tmp','TEMP':'/tmp',
                 'BENCH_PRIVATE_EVAL':'1','BENCH_BENCH_DIR':'/ws/bench','BENCH_CODE_DIR':'/ws/code',
                 'BENCH_DATA_DIR':'/ws/resources','BENCH_MODEL_DIR':'/ws/model','BENCH_CPUS':str(worker_cpus),
                 'BENCH_SEED':str(self.manifest['algorithm_seed']),'PYTHONDONTWRITEBYTECODE':'1',
                 'PYTHONSAFEPATH':'1','PYTHONPATH':'/ws/deps/site-packages:/ws/bench:/ws/code',
                 'OMP_NUM_THREADS':str(worker_cpus),'OPENBLAS_NUM_THREADS':str(worker_cpus),
                 'MKL_NUM_THREADS':str(worker_cpus),'NUMEXPR_NUM_THREADS':str(worker_cpus),'PJRT_NPROC':str(worker_cpus)}
        if broker: fixed['BENCH_RPC_SOCKET'] = '/run/private-eval/rpc.sock'
        values = {**extra,**fixed}
        if env and 'BENCH_SEED' in env: values['BENCH_SEED'] = str(env['BENCH_SEED'])
        cmd += ['--clearenv']
        for key,value in values.items(): cmd += ['--setenv',key,str(value)]
        cmd += ['--chdir','/ws/code','--','bash','/ws/code/policy.sh','--model','/ws/model',
                '--input','/ws/input','--output','/ws/output/pred.out']
        try:
            caps = []
            overall_remaining = remaining_time()
            if overall_remaining is not None: caps.append(overall_remaining)
            if self.timeout_s is not None:
                caps.append(self.timeout_s - (time.monotonic()-self.started))
            if timeout_s is not None: caps.append(float(timeout_s))
            cap = min(caps) if caps else None
            if cap is not None and (not math.isfinite(cap) or cap <= 0):
                raise TimeoutError('task replay time budget exhausted')
        except Exception:
            if broker: broker.close()
            raise
        started = time.monotonic()
        try:
            with (work/'stdout.log').open('wb') as stdout,(work/'stderr.log').open('wb') as stderr:
                process = subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                           close_fds=True,start_new_session=True,env={'PATH':'/usr/bin:/bin'})
                try: code = process.wait(timeout=cap)
                except subprocess.TimeoutExpired: raise TimeoutError('isolated policy exceeded time limit')
                finally:
                    with contextlib.suppress(ProcessLookupError): os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=5)
            if broker and broker.error: raise ValueError('private oracle protocol violation ('+broker.error+')')
            if code:
                error = (work/'stderr.log').read_text(errors='replace')[-4000:]
                raise RuntimeError('isolated policy failed rc=%s: %s'%(code,error))
            result = work/'prediction.checked'; _safe_output(output/'pred.out',result)
            return {'output_path':str(result),'elapsed_s':time.monotonic()-started,'returncode':code,
                    'rpc_calls':broker.calls if broker else 0,
                    'stdout_path':str(work/'stdout.log'),'stderr_path':str(work/'stderr.log')}
        finally:
            if broker: broker.close()

def main():
    p = argparse.ArgumentParser()
    for name in ('code','model','output','task-id'): p.add_argument('--'+name,required=True)
    p.add_argument('--resources'); p.add_argument('--no-dependencies',action='store_true')
    a = p.parse_args()
    export_artifact(a.code,a.model,a.output,a.task_id,a.resources,not a.no_dependencies)

if __name__ == '__main__': main()

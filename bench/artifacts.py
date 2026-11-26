"""Opt-in candidate file versions and continuous native session preservation.

Only the candidate workspace and native session logs are archived. Operator or
sandbox authentication homes, package caches and read-only datasets are excluded.
"""
import ctypes
import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time


class SolutionRecorder:
    CLOSE_WRITE, MOVED_FROM, MOVED_TO, CREATE, DELETE = 8, 64, 128, 256, 512
    ISDIR, OVERFLOW = 0x40000000, 0x4000

    def __init__(self, workspace, native_sessions, destination):
        self.workspace, self.native = Path(workspace), Path(native_sessions)
        self.destination = Path(destination)
        self.solutions = self.destination/'solutions'
        self.objects = self.solutions/'objects'
        self.objects.mkdir(parents=True,exist_ok=True)
        self.index = self.solutions/'versions.jsonl'
        self.lock=threading.Lock();self.stopped=threading.Event();self.watches={}
        self.seen={};self.versions=0;self.errors=[];self.started=time.time();self.threads=[]
        self.event_batches=queue.SimpleQueue();self.reader_done=threading.Event()
        self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.inotify_init1.argtypes=[ctypes.c_int]
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd < 0:raise OSError(ctypes.get_errno(),'inotify_init1')

    def included(self,path):
        rel=path.relative_to(self.workspace)
        return bool(rel.parts) and rel.parts[0] not in ('data','bench','paper') and '.git' not in rel.parts and '__pycache__' not in rel.parts

    def record(self,value):
        with self.lock:
            with self.index.open('a',encoding='utf-8') as out:
                out.write(json.dumps({'observed_ns':time.time_ns(),**value},ensure_ascii=False)+'\n')

    def error(self,where,exc):
        item={'event':'capture_error','path':str(where),'error':str(exc)[:500]}
        self.errors.append(item);self.record(item)

    def capture(self,path,event):
        if not self.included(path):return
        relative=str(path.relative_to(self.workspace))
        try:
            st=path.lstat()
            if stat.S_ISLNK(st.st_mode):
                target=os.readlink(path)
                if self.seen.get(relative)!=('symlink',target):
                    self.record({'event':event,'path':relative,'kind':'symlink','target':target})
                    self.seen[relative]=('symlink',target)
                return
            if not stat.S_ISREG(st.st_mode):return
            signature=(st.st_ino,st.st_size,st.st_mtime_ns)
            # A close/move event proves a write may have happened even when a
            # same-size edit lands within the filesystem's timestamp tick.
            if self.seen.get(relative)==signature and event not in ('close_write','moved_to'):return
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            temp_fd,temp=tempfile.mkstemp(prefix='.capture-',dir=self.objects)
            digest=hashlib.sha256();size=0
            try:
                with os.fdopen(fd,'rb') as src,os.fdopen(temp_fd,'wb') as dst:
                    before=os.fstat(src.fileno())
                    while chunk:=src.read(2**20):digest.update(chunk);dst.write(chunk);size+=len(chunk)
                    after=os.fstat(src.fileno())
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    self.record({'event':'changed_during_capture','path':relative})
                    return
                sha=digest.hexdigest();obj=self.objects/sha
                if obj.exists():os.unlink(temp)
                else:os.replace(temp,obj)
                self.record({'event':event,'path':relative,'kind':'file','sha256':sha,'bytes':size,
                             'mtime_ns':after.st_mtime_ns,'mode':stat.S_IMODE(after.st_mode)})
                self.seen[relative]=(after.st_ino,after.st_size,after.st_mtime_ns)
                self.versions+=1
            finally:
                if os.path.exists(temp):os.unlink(temp)
        except FileNotFoundError:
            self.record({'event':'path_moved_or_deleted_before_capture','path':relative,'trigger':event})
        except PermissionError:
            # Installers create 0600 temporary metadata as the sandbox uid.
            # Preserve readable contents via a narrowly scoped read-only helper.
            try:self._capture_private(path,event)
            except Exception as exc:self.error(relative,exc)
        except OSError as exc:self.error(relative,exc)

    def _capture_private(self,path,event):
        relative=str(path.relative_to(self.workspace))
        fd,temp=tempfile.mkstemp(prefix='.capture-',dir=self.objects)
        command=['sudo','-n',sys.executable,'-I','-S',str(Path(__file__).with_name('artifact_file_reader.py')),
                 str(self.workspace),relative]
        digest=hashlib.sha256();size=0;metadata=None;gone=False;changed=False
        try:
            with os.fdopen(fd,'wb') as dst, subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
                for line in child.stdout:
                    row=json.loads(line)
                    if 'data' in row:
                        data=base64.b64decode(row['data'],validate=True)
                        dst.write(data);digest.update(data);size+=len(data)
                    metadata=row.get('metadata',metadata);gone=gone or row.get('gone',False);changed=changed or row.get('changed',False)
                error=child.stderr.read()
                if child.wait()!=0:raise RuntimeError('Candidate reader failed: '+error.decode(errors='replace')[-300:])
            if gone:
                self.record({'event':'path_moved_or_deleted_before_capture','path':relative,'trigger':event});return
            if changed:
                self.record({'event':'changed_during_capture','path':relative});return
            if metadata is None or size!=metadata['size']:raise RuntimeError('Incomplete candidate file capture')
            sha=digest.hexdigest();obj=self.objects/sha
            if not obj.exists():os.replace(temp,obj)
            self.record({'event':event,'path':relative,'kind':'file','sha256':sha,'bytes':size,
                'mtime_ns':metadata['mtime_ns'],'mode':metadata['mode'],'reader':'private_candidate_file'})
            self.seen[relative]=(metadata['inode'],size,metadata['mtime_ns']);self.versions+=1
        finally:
            if os.path.exists(temp):os.unlink(temp)

    def watch_tree(self,root,event='new_directory'):
        for directory,dirs,files in os.walk(root,followlinks=False):
            d=Path(directory)
            dirs[:]=[name for name in dirs if self.included(d/name) and not (d/name).is_symlink()]
            wd=self.libc.inotify_add_watch(self.fd,os.fsencode(d),self.CLOSE_WRITE|self.MOVED_FROM|self.MOVED_TO|self.CREATE|self.DELETE)
            if wd<0:
                code=ctypes.get_errno()
                if code in (errno.ENOENT,errno.ENOTDIR):
                    # Package installers and pytest remove temporary directories
                    # while the watcher is registering them. Keep the evidence
                    # of that race and continue watching the rest of the tree.
                    self.record({'event':'path_moved_or_deleted_before_watch',
                                 'path':str(d.relative_to(self.workspace)),'trigger':'inotify_add_watch'})
                    continue
                raise OSError(code,'inotify_add_watch '+str(d))
            self.watches[wd]=d
            for name in files:self.capture(d/name,event)
            try:names=os.listdir(d)
            except (FileNotFoundError,NotADirectoryError):
                self.record({'event':'path_moved_or_deleted_before_watch',
                             'path':str(d.relative_to(self.workspace)),'trigger':'listdir'})
                continue
            for name in names:
                if (d/name).is_symlink():self.capture(d/name,'symlink')

    def snapshot(self):
        for directory,dirs,files in os.walk(self.workspace,followlinks=False):
            d=Path(directory)
            dirs[:]=[n for n in dirs if self.included(d/n) and not (d/n).is_symlink()]
            for name in files:self.capture(d/name,'final')

    def read_events(self):
        # Hashing a large solution or snapshotting an extracted directory must
        # not block draining the small kernel queue. Retain every event batch;
        # the single consumer still serializes captures and directory watches.
        try:
            while True:
                if not select.select([self.fd],[],[],.2)[0]:
                    if self.stopped.is_set():break
                    continue
                try:events=os.read(self.fd,262144)
                except BlockingIOError:continue
                if events:self.event_batches.put(events)
        except Exception as exc:self.error('inotify_reader',exc)
        finally:self.reader_done.set()

    def watch(self):
        try:
            while True:
                try:events=self.event_batches.get(timeout=.2)
                except queue.Empty:
                    if self.reader_done.is_set():break
                    continue
                offset=0
                while offset<len(events):
                    wd,mask,cookie,length=struct.unpack_from('iIII',events,offset);offset+=16
                    name=os.fsdecode(events[offset:offset+length].split(b'\0',1)[0]);offset+=length
                    if mask&self.OVERFLOW:
                        self.error('inotify',RuntimeError('event queue overflow'));self.snapshot();continue
                    if wd not in self.watches:continue
                    path=self.watches[wd]/name
                    if not self.included(path):continue
                    if mask&self.ISDIR:
                        if mask&(self.CREATE|self.MOVED_TO) and path.is_dir() and not path.is_symlink():self.watch_tree(path)
                    elif mask&(self.CLOSE_WRITE|self.MOVED_TO):self.capture(path,'close_write' if mask&self.CLOSE_WRITE else 'moved_to')
                    elif mask&(self.DELETE|self.MOVED_FROM):
                        self.record({'event':'deleted' if mask&self.DELETE else 'moved_from','path':str(path.relative_to(self.workspace))})
                        self.seen.pop(str(path.relative_to(self.workspace)),None)
        except Exception as exc:self.error('watcher',exc)

    def mirror_native(self):
        try:self._mirror_native_direct()
        except PermissionError:self._mirror_native_private()

    def _mirror_native_private(self):
        target=self.destination/'trace/kimi/sessions';target.mkdir(parents=True,exist_ok=True)
        offsets={str(p.relative_to(target)):p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink()}
        command=['sudo','-n',sys.executable,'-I',str(Path(__file__).with_name('artifact_reader.py')),str(self.native.parent.parent)]
        with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
            child.stdin.write((json.dumps(offsets)+'\n').encode());child.stdin.close()
            for line in child.stdout:
                row=json.loads(line);relative=Path(row['path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe native archive path')
                dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
                data=base64.b64decode(row['data'],validate=True)
                if row['state']:
                    temp=dst.with_suffix('.live.tmp');temp.write_bytes(data);os.replace(temp,dst)
                else:
                    if row.get('reset') and dst.exists():os.replace(dst,dst.with_name(dst.name+f'.previous-{time.time_ns()}'))
                    offset=dst.stat().st_size if dst.exists() else 0
                    if offset!=row['offset']:raise ValueError('Native archive offset mismatch')
                    with dst.open('ab') as out:out.write(data)
            child.stdout.close()
            error=child.stderr.read();child.stderr.close()
            if child.wait()!=0:raise RuntimeError('Native session reader failed: '+error.decode(errors='replace')[-300:])

    def _mirror_native_direct(self):
        if not self.native.exists():return
        target=self.destination/'trace/kimi/sessions'
        for src in self.native.rglob('*'):
            if src.is_symlink() or not src.is_file() or not (src.name in ('wire.jsonl','state.json') or src.suffix=='.log'):continue
            dst=target/src.relative_to(self.native);dst.parent.mkdir(parents=True,exist_ok=True)
            try:
                if src.name=='state.json':
                    temp=dst.with_suffix('.live.tmp');shutil.copyfile(src,temp);os.replace(temp,dst)
                else:
                    offset=dst.stat().st_size if dst.exists() else 0
                    if src.stat().st_size<offset:
                        os.replace(dst,dst.with_name(dst.name+f'.previous-{time.time_ns()}'));offset=0
                    with src.open('rb') as a,dst.open('ab') as b:
                        a.seek(offset);shutil.copyfileobj(a,b,2**20)
            except FileNotFoundError:pass
            except PermissionError:
                # Native state may be atomically replaced with mode 0600 while
                # the directory stays searchable. Use the same read-only bridge
                # as for a wholly private HOME instead of pausing the Agent.
                raise
            except OSError as exc:self.error('native/'+str(src.relative_to(self.native)),exc)

    def mirror(self):
        while not self.stopped.wait(2):
            try:self.mirror_native()
            except Exception as exc:self.error('native_mirror',exc)

    def start(self):
        reader=threading.Thread(target=self.read_events,daemon=True)
        self.threads=[reader];reader.start()
        try:self.watch_tree(self.workspace,event='initial')
        except BaseException:
            self.stopped.set();reader.join();os.close(self.fd)
            raise
        consumers=[threading.Thread(target=self.watch,daemon=True),threading.Thread(target=self.mirror,daemon=True)]
        self.threads.extend(consumers)
        for t in consumers:t.start()
        return self

    def stop(self):
        self.stopped.set()
        for t in self.threads:t.join()
        os.close(self.fd)
        self.snapshot();self.mirror_native()
        out={'started_ts':self.started,'finished_ts':time.time(),'file_versions':self.versions,
             'errors':self.errors,'status':'complete' if not self.errors else 'needs_review',
             'workspace':'candidate workspace excluding read-only data/paper/bench, .git and __pycache__',
             'native_sessions':'wire.jsonl, state.json and native logs; authentication files excluded',
             'index':'solutions/versions.jsonl','objects':'solutions/objects/<sha256>',
             'capture':'inotify close-write/move events plus initial/final snapshots; transient rename/deletion races are retained in the index'}
        (self.solutions/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
        return out

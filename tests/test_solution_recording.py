import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
import os
import subprocess
import ctypes
import errno
import shutil
import threading
from unittest.mock import patch

from bench.artifacts import SolutionRecorder
from bench.run import pick_final


class SolutionRecordingTests(unittest.TestCase):
    def test_slow_capture_does_not_overflow_kernel_event_queue(self):
        limit=int(Path('/proc/sys/fs/inotify/max_queued_events').read_text())
        if limit>65536:self.skipTest('Host queue too large for a bounded burst test')
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);ws=root/'ws';(ws/'code').mkdir(parents=True)
            recorder=SolutionRecorder(ws,root/'native',root/'archive')
            blocked=threading.Event();release=threading.Event();capture=recorder.capture
            def slow_capture(path,event):
                if path.name=='hold.py' and event=='close_write':
                    blocked.set();release.wait(20)
                return capture(path,event)
            recorder.capture=slow_capture;recorder.start()
            try:
                (ws/'code/hold.py').write_text('slow solution copy')
                self.assertTrue(blocked.wait(3))
                # Each file produces create, close-write and delete events.
                # The consumer remains blocked throughout this whole burst.
                for n in range(limit//3+2048):
                    p=ws/'code'/f'burst-{n}.py';p.write_text('x');p.unlink()
                self.assertFalse(release.is_set())
                self.assertFalse(recorder.reader_done.is_set())
            finally:
                release.set();result=recorder.stop()
            self.assertEqual(result['errors'],[])
            self.assertTrue(any(p.read_bytes()==b'slow solution copy' for p in recorder.objects.iterdir()))

    def test_private_candidate_file_is_archived_without_changing_its_permissions(self):
        if subprocess.run(['sudo','-n','true'],capture_output=True).returncode:
            self.skipTest('sudo unavailable')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);root.chmod(0o755);ws=root/'ws';(ws/'code').mkdir(parents=True)
            path=ws/'code/private_candidate.py';path.write_text('saved solution');path.chmod(0o600)
            subprocess.run(['sudo','-n','chown','3000998:3000998',str(path)],check=True)
            recorder=SolutionRecorder(ws,root/'native',root/'archive')
            try:
                recorder.capture(path,'close_write')
                self.assertEqual(recorder.errors,[])
                self.assertTrue(any(p.read_bytes()==b'saved solution' for p in recorder.objects.iterdir()))
                self.assertEqual(path.stat().st_mode&0o777,0o600)
                self.assertEqual(path.stat().st_uid,3000998)
            finally:
                recorder.stop()
                subprocess.run(['sudo','-n','chown',f'{os.getuid()}:{os.getgid()}',str(path)],check=True)

    def test_private_candidate_reader_rejects_parent_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ws=root/'ws';ws.mkdir();secret=root/'secret';secret.mkdir()
            (secret/'key').write_text('must-not-archive');(ws/'code').symlink_to(secret)
            reader=Path(__file__).resolve().parents[1]/'bench/artifact_file_reader.py'
            result=subprocess.run(['python3',str(reader),str(ws),'code/key'],capture_output=True,text=True)
            self.assertNotIn('data',result.stdout)
            self.assertNotIn('must-not-archive',result.stdout)

    def test_write_event_preserves_same_size_same_timestamp_edit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ws=root/'ws';(ws/'code').mkdir(parents=True)
            path=ws/'code/main.py';path.write_text('version one');before=path.stat()
            recorder=SolutionRecorder(ws,root/'native',root/'archive')
            try:
                recorder.capture(path,'initial')
                path.write_text('version two')
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
                recorder.capture(path,'close_write')
            finally:recorder.stop()
            objects=[p.read_bytes() for p in recorder.objects.iterdir()]
            self.assertIn(b'version one',objects)
            self.assertIn(b'version two',objects)

    def test_directory_deletion_during_watch_registration_is_nonfatal(self):
        for stage in ('inotify_add_watch','listdir'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as d:
                root=Path(d);ws=root/'ws';transient=ws/'code/temporary';transient.mkdir(parents=True)
                (ws/'code/main.py').write_text('solution')
                recorder=SolutionRecorder(ws,root/'native',root/'archive')
                if stage=='inotify_add_watch':
                    original=recorder.libc.inotify_add_watch
                    def race(fd,path,mask):
                        if Path(os.fsdecode(path))==transient:shutil.rmtree(transient)
                        return original(fd,path,mask)
                    interception=patch.object(recorder.libc,'inotify_add_watch',side_effect=race)
                else:
                    original=os.listdir
                    def race(path):
                        if Path(path)==transient:shutil.rmtree(transient)
                        return original(path)
                    interception=patch('bench.artifacts.os.listdir',side_effect=race)
                with interception:recorder.watch_tree(ws)
                (ws/'code/later.py').write_text('next solution')
                recorder.watch_tree(ws)
                info=recorder.stop()
                self.assertEqual(info['errors'],[])
                rows=[json.loads(s) for s in recorder.index.read_text().splitlines()]
                self.assertTrue(any(r.get('trigger')==stage and r['event']=='path_moved_or_deleted_before_watch' for r in rows))
                self.assertTrue(any(r.get('path')=='code/later.py' and r.get('kind')=='file' for r in rows))

    def test_watch_resource_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ws=root/'ws';ws.mkdir()
            recorder=SolutionRecorder(ws,root/'native',root/'archive')
            def fail(*args):ctypes.set_errno(errno.ENOSPC);return -1
            try:
                with patch.object(recorder.libc,'inotify_add_watch',side_effect=fail):
                    with self.assertRaises(OSError) as caught:recorder.watch_tree(ws)
                self.assertEqual(caught.exception.errno,errno.ENOSPC)
            finally:recorder.stop()

    def test_private_state_file_in_searchable_directory_uses_bridge(self):
        if subprocess.run(['sudo','-n','true'],capture_output=True).returncode:
            self.skipTest('sudo unavailable')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);root.chmod(0o755);ws=root/'ws';(ws/'code').mkdir(parents=True)
            home=root/'home';native=home/'.kimi-code/sessions/wd/session';native.mkdir(parents=True)
            state=native/'state.json';state.write_text('{"id":"session","agents":{"main":{}}}')
            state.chmod(0o600)
            subprocess.run(['sudo','-n','chown','3000998:3000998',str(state)],check=True)
            try:
                dest=root/'archive';r=SolutionRecorder(ws,home/'.kimi-code/sessions',dest).start()
                try:r.mirror_native()
                finally:result=r.stop()
                self.assertEqual(json.loads((dest/'trace/kimi/sessions/wd/session/state.json').read_text())['id'],'session')
                self.assertEqual(result['errors'],[])
            finally:
                subprocess.run(['sudo','-n','chown',f'{os.getuid()}:{os.getgid()}',str(state)],check=True)

    def test_private_native_session_bridge_ignores_symlinks(self):
        if subprocess.run(['sudo','-n','true'],capture_output=True).returncode:
            self.skipTest('sudo unavailable')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);root.chmod(0o755);ws=root/'ws';(ws/'code').mkdir(parents=True)
            home=root/'home';native=home/'.kimi-code/sessions/wd/session/agents/main';native.mkdir(parents=True)
            (native/'wire.jsonl').write_text('{"type":"usage.record"}\n')
            secret=root/'private.log';secret.write_text('must-not-archive')
            (native/'outside.log').symlink_to(secret)
            for directory in (home,home/'.kimi-code',home/'.kimi-code/sessions'):directory.chmod(0o700)
            subprocess.run(['sudo','-n','chown','-R','--no-dereference','3000999:3000999',str(home)],check=True)
            try:
                dest=root/'archive';r=SolutionRecorder(ws,home/'.kimi-code/sessions',dest).start()
                try:r.mirror_native()
                finally:r.stop()
                archived=dest/'trace/kimi/sessions/wd/session/agents/main'
                self.assertEqual((archived/'wire.jsonl').read_text(),'{"type":"usage.record"}\n')
                self.assertFalse((archived/'outside.log').exists())
            finally:
                subprocess.run(['sudo','-n','chown','-R','--no-dereference',f'{os.getuid()}:{os.getgid()}',str(home)],check=True)

    def test_last_valid_submission_overrides_earlier_final(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,final,ts,ok in [(1,True,1,True),(2,False,2,True),(3,True,3,False),(4,True,11,True)]:
                data=f'candidate-{n}'.encode()
                (root/f'{n:03d}.patch').write_bytes(data)
                (root/f'{n:03d}.json').write_text(json.dumps({'n':n,'final':final,'ts':ts,'ok':ok,'source':'submit','bytes':len(data),'patch_sha256':hashlib.sha256(data).hexdigest()}))
            self.assertEqual(pick_final(root,10)[0]['n'],1)
            chosen,all_submissions=pick_final(root,10,prefer_final=False)
            self.assertEqual(chosen['n'],2)
            self.assertEqual(len(all_submissions),4)
            (root/'002.patch').write_bytes(b'corrupted')
            self.assertEqual(pick_final(root,10,prefer_final=False)[0]['n'],1)

    def test_versions_native_trace_and_no_credential_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ws=root/'ws';(ws/'code').mkdir(parents=True)
            native=root/'home/.kimi-code/sessions';native.mkdir(parents=True)
            creds=root/'home/.kimi-code/credentials';creds.mkdir();(creds/'secret.json').write_text('private-token')
            (ws/'code/main.py').write_text('version one')
            (ws/'code/outside').symlink_to(creds/'secret.json')
            dest=root/'archive';r=SolutionRecorder(ws,native,dest).start()
            try:
                (ws/'code/main.py').write_text('version two')
                until=time.monotonic()+3
                while r.versions<2 and time.monotonic()<until:time.sleep(.02)
                (native/'wire.jsonl').write_text('{"type":"tool.call"}\n')
                r.mirror_native()
                with (native/'wire.jsonl').open('a') as f:f.write('{"type":"usage.record"}\n')
            finally:info=r.stop()
            objects=[p.read_bytes() for p in (dest/'solutions/objects').iterdir()]
            self.assertIn(b'version one',objects);self.assertIn(b'version two',objects)
            self.assertNotIn(b'private-token',objects)
            self.assertEqual((dest/'trace/kimi/sessions/wire.jsonl').read_text().count('\n'),2)
            self.assertFalse(list(dest.rglob('secret.json')))
            self.assertEqual(info['status'],'complete')


if __name__=='__main__':unittest.main()

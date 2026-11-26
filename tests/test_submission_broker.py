import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bench import submission


PATCH = b"diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-old\n+new\n"
LIMITS = {"max_bytes": 2 * 1024 * 1024, "forbidden_new_ext": [".npy"]}


class SubmissionBrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="arb-submit-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.socket_path = self.root / "submit.sock"
        self.storage = self.root / "sealed"

    def broker(self, deadline=None, limits=None):
        service = submission.SubmissionBroker(
            self.socket_path, self.storage,
            time.time() + 10 if deadline is None else deadline,
            LIMITS if limits is None else limits).start()
        self.addCleanup(service.stop)
        return service

    def response(self, conn):
        end = time.monotonic() + 2
        length, = submission._RESPONSE_HEADER.unpack(
            submission._read_exact(conn, submission._RESPONSE_HEADER.size, end))
        return json.loads(submission._read_exact(conn, length, end))

    def raw_submit(self, metadata, body=PATCH):
        payload = json.dumps(metadata).encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(2)
            conn.connect(str(self.socket_path))
            try:
                conn.sendall(submission._HEADER.pack(submission._MAGIC, len(payload), len(body))
                             + payload + body)
            except BrokenPipeError:
                pass  # An already expired broker can reply before consuming input.
            return self.response(conn)

    def assert_empty_storage(self):
        self.assertEqual(list(self.storage.iterdir()), [])

    def test_valid_submission_has_host_metadata_and_sealed_files(self):
        service = self.broker()
        begin = time.time()
        with patch("bench.patch.git", side_effect=AssertionError("broker must not run git")):
            record = submission.client_submit(self.socket_path, PATCH, final=True,
                                               msg="candidate", self_reported_val={"accuracy": 0.8})
        self.assertTrue(record["ok"])
        self.assertEqual(record["n"], 1)
        self.assertEqual(record["source"], "submit")
        self.assertEqual(record["patch_sha256"], hashlib.sha256(PATCH).hexdigest())
        self.assertEqual((record["bytes"], record["files"]), (len(PATCH), 1))
        self.assertEqual(record["file_list"], ["model.py"])
        self.assertEqual(record["problems"], [])
        self.assertTrue(record["final"])
        self.assertEqual(record["msg"], "candidate")
        self.assertEqual(record["self_reported_val"], {"accuracy": 0.8})
        self.assertLessEqual(begin, record["received_ts"])
        self.assertLessEqual(record["received_ts"], record["validated_ts"])
        self.assertLessEqual(record["validated_ts"], record["ts"])
        self.assertLessEqual(record["ts"], service.deadline_ts)
        self.assertEqual((self.storage / "001.patch").read_bytes(), PATCH)
        self.assertEqual(json.loads((self.storage / "001.json").read_text()), record)
        for path in self.storage.iterdir():
            self.assertEqual(path.stat().st_mode & 0o222, 0)

    def test_empty_patch_is_an_explicit_valid_baseline_submission(self):
        self.broker()
        record = submission.client_submit(self.socket_path, b"")
        self.assertTrue(record["ok"])
        self.assertEqual((record["bytes"], record["files"]), (0, 0))
        self.assertEqual((self.storage / "001.patch").read_bytes(), b"")

    def test_fragmented_request_is_received_completely(self):
        self.broker()
        metadata = b"{}"
        frame = submission._HEADER.pack(submission._MAGIC, len(metadata), len(PATCH)) + metadata + PATCH
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(2)
            conn.connect(str(self.socket_path))
            for value in frame:
                conn.sendall(bytes([value]))
            record = self.response(conn)
        self.assertTrue(record["ok"])
        self.assertEqual((self.storage / "001.patch").read_bytes(), PATCH)

    def test_expired_and_backdated_requests_are_rejected(self):
        self.broker(deadline=time.time() - 1)
        record = self.raw_submit({"ts": 0, "ok": True, "n": 1, "source": "submit"})
        self.assertFalse(record["ok"])
        self.assertEqual(record["stage"], "deadline")
        self.assertNotIn("n", record)
        self.assert_empty_storage()

    def test_host_ignores_forged_validity_count_digest_and_timestamp(self):
        self.broker()
        begin = time.time()
        record = self.raw_submit({"ts": 0, "ok": False, "n": 999, "source": "auto",
                                  "patch_sha256": "forged", "bytes": 0, "files": 99,
                                  "file_list": ["forged"], "final": True, "msg": "diagnostic"})
        self.assertTrue(record["ok"])
        self.assertEqual(record["n"], 1)
        self.assertGreaterEqual(record["ts"], begin)
        self.assertEqual(record["source"], "submit")
        self.assertEqual(record["patch_sha256"], hashlib.sha256(PATCH).hexdigest())
        self.assertEqual((record["bytes"], record["files"]), (len(PATCH), 1))
        self.assertEqual(record["file_list"], ["model.py"])

    def test_invalid_patch_cannot_forge_a_valid_record(self):
        self.broker()
        unsafe = b"diff --git a/../secret b/../secret\n--- a/../secret\n+++ b/../secret\n+bad\n"
        record = self.raw_submit({"ok": True, "problems": [], "ts": 0}, unsafe)
        self.assertFalse(record["ok"])
        self.assertIn("unsafe path", " ".join(record["problems"]))
        self.assertEqual(record["n"], 1)
        self.assertEqual((self.storage / "001.patch").read_bytes(), unsafe)
        binary = b"diff --git a/a b/a\nGIT binary patch\nliteral 1\nx\n"
        record = submission.client_submit(self.socket_path, binary)
        self.assertFalse(record["ok"])
        self.assertIn("binary content", " ".join(record["problems"]))
        self.assertEqual(record["n"], 2)

    def test_body_must_be_complete_before_host_deadline(self):
        host_now = [999.0]
        body_started = threading.Event()
        original_read = submission._read_exact

        def observe_body(conn, size, io_end, deadline_ts=None):
            if size == len(PATCH) and deadline_ts is not None:
                body_started.set()
            return original_read(conn, size, io_end, deadline_ts)

        with patch("bench.submission.time.time", side_effect=lambda: host_now[0]):
            self.broker(deadline=1000)
            with patch("bench.submission._read_exact", side_effect=observe_body):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(2)
                    conn.connect(str(self.socket_path))
                    metadata = b'{"ts":0}'
                    conn.sendall(submission._HEADER.pack(submission._MAGIC, len(metadata), len(PATCH))
                                 + metadata + PATCH[:-1])
                    self.assertTrue(body_started.wait(2))
                    host_now[0] = 1001.0
                    conn.sendall(PATCH[-1:])
                    record = self.response(conn)
            self.assertEqual(record["stage"], "deadline")
            self.assertFalse(record["ok"])
            self.assert_empty_storage()

    def test_validation_must_finish_before_host_deadline(self):
        host_now = [999.0]
        validating, release = threading.Event(), threading.Event()
        original_check = submission.patchmod.check_patch

        def slow_check(*args, **kwargs):
            validating.set()
            if not release.wait(2):
                raise RuntimeError("test validation release timed out")
            return original_check(*args, **kwargs)

        with patch("bench.submission.time.time", side_effect=lambda: host_now[0]):
            self.broker(deadline=1000)
            with patch("bench.submission.patchmod.check_patch", side_effect=slow_check):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(submission.client_submit, self.socket_path, PATCH)
                    try:
                        self.assertTrue(validating.wait(2))
                        host_now[0] = 1001.0
                    finally:
                        release.set()
                    record = future.result(timeout=3)
            self.assertEqual(record["stage"], "deadline")
            self.assertFalse(record["ok"])
            self.assert_empty_storage()

    def test_expiry_during_sealing_removes_partial_artifacts(self):
        host_now = [999.0]
        with patch("bench.submission.time.time", side_effect=lambda: host_now[0]):
            service = self.broker(deadline=1000)
            original_write = service._write_pending
            calls = []

            def expire_after_write(content):
                path = original_write(content)
                calls.append(path)
                if len(calls) == 2:
                    host_now[0] = 1001.0
                return path

            with patch.object(service, "_write_pending", side_effect=expire_after_write):
                record = submission.client_submit(self.socket_path, PATCH)
            self.assertEqual(record["stage"], "deadline")
            self.assertFalse(record["ok"])
            self.assert_empty_storage()

    def test_expiry_after_publication_rolls_back_both_files(self):
        host_now = [999.0]
        with patch("bench.submission.time.time", side_effect=lambda: host_now[0]):
            self.broker(deadline=1000)
            original_fsync = os.fsync

            def expire_on_directory_sync(fd):
                original_fsync(fd)
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    self.assertTrue((self.storage / "001.patch").is_file())
                    self.assertTrue((self.storage / "001.json").is_file())
                    host_now[0] = 1001.0

            with patch("bench.submission.os.fsync", side_effect=expire_on_directory_sync):
                record = submission.client_submit(self.socket_path, PATCH)
            self.assertEqual(record["stage"], "deadline")
            self.assertFalse(record["ok"])
            self.assert_empty_storage()

    def test_client_buffer_changes_do_not_change_sealed_content(self):
        self.broker()
        validating, release = threading.Event(), threading.Event()
        original_check = submission.patchmod.check_patch
        buffer = bytearray(PATCH)

        def delayed_check(*args, **kwargs):
            validating.set()
            if not release.wait(2):
                raise RuntimeError("test validation release timed out")
            return original_check(*args, **kwargs)

        with patch("bench.submission.patchmod.check_patch", side_effect=delayed_check):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(submission.client_submit, self.socket_path, buffer)
                try:
                    self.assertTrue(validating.wait(2))
                    buffer[:] = b"modified after receipt"
                finally:
                    release.set()
                record = future.result(timeout=3)
        self.assertTrue(record["ok"])
        self.assertEqual((self.storage / "001.patch").read_bytes(), PATCH)
        self.assertEqual(record["patch_sha256"], hashlib.sha256(PATCH).hexdigest())

    def test_concurrent_submissions_have_unique_contiguous_numbers(self):
        self.broker()
        barrier = threading.Barrier(4)
        original_check = submission.patchmod.check_patch

        def concurrent_check(*args, **kwargs):
            barrier.wait(timeout=3)
            return original_check(*args, **kwargs)

        patches = [PATCH.replace(b"+new", f"+candidate_{i}".encode()) for i in range(4)]
        with patch("bench.submission.patchmod.check_patch", side_effect=concurrent_check):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(submission.client_submit, self.socket_path, body) for body in patches]
                records = [f.result(timeout=4) for f in futures]
        self.assertEqual({r["n"] for r in records}, {1, 2, 3, 4})
        for body, record in zip(patches, records):
            self.assertTrue(record["ok"])
            stem = f'{record["n"]:03d}'
            self.assertEqual((self.storage / f"{stem}.patch").read_bytes(), body)
            self.assertEqual(json.loads((self.storage / f"{stem}.json").read_text()), record)
        self.assertEqual(len(list(self.storage.iterdir())), 8)

    def test_restart_preserves_existing_records_and_continues_numbering(self):
        service = self.broker()
        first = submission.client_submit(self.socket_path, PATCH)
        service.stop()
        self.assertFalse(self.socket_path.exists())
        service = self.broker()
        second = submission.client_submit(self.socket_path, b"")
        self.assertEqual((first["n"], second["n"]), (1, 2))
        self.assertEqual((self.storage / "001.patch").read_bytes(), PATCH)

    def test_bad_header_and_oversized_lengths_reject_before_reading_body(self):
        self.broker(limits={"max_bytes": 100})
        headers = [(b"BAD!", 2, 0), (submission._MAGIC, submission._MAX_METADATA_BYTES + 1, 0),
                   (submission._MAGIC, 2, 101), (submission._MAGIC, 0, 0)]
        with patch("bench.submission.patchmod.check_patch") as checker:
            for magic, metadata_size, patch_size in headers:
                with self.subTest(header=(magic, metadata_size, patch_size)):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                        conn.settimeout(2)
                        conn.connect(str(self.socket_path))
                        conn.sendall(submission._HEADER.pack(magic, metadata_size, patch_size))
                        record = self.response(conn)
                    self.assertFalse(record["ok"])
                    self.assertEqual(record["stage"], "protocol")
            checker.assert_not_called()
        self.assert_empty_storage()

    def test_stop_interrupts_incomplete_request(self):
        service = self.broker()
        reading = threading.Event()
        original_read = submission._read_exact

        def observe_read(*args, **kwargs):
            reading.set()
            return original_read(*args, **kwargs)

        with patch("bench.submission._read_exact", side_effect=observe_read):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.connect(str(self.socket_path))
                conn.sendall(b"A")
                self.assertTrue(reading.wait(2))
                before = time.monotonic()
                service.stop()
                self.assertLess(time.monotonic() - before, 1)
        self.assertFalse(service._accept_thread.is_alive())
        self.assertEqual(service._workers, set())
        self.assert_empty_storage()

    def test_stop_waits_for_validation_and_prevents_later_writes(self):
        service = self.broker()
        validating, release, stopped = threading.Event(), threading.Event(), threading.Event()
        original_check = submission.patchmod.check_patch

        def delayed_check(*args, **kwargs):
            validating.set()
            if not release.wait(2):
                raise RuntimeError("test validation release timed out")
            return original_check(*args, **kwargs)

        def stop():
            service.stop()
            stopped.set()

        with patch("bench.submission.patchmod.check_patch", side_effect=delayed_check):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                request = pool.submit(submission.client_submit, self.socket_path, PATCH)
                self.assertTrue(validating.wait(2))
                stopping = pool.submit(stop)
                try:
                    self.assertFalse(stopped.wait(0.03))
                finally:
                    release.set()
                stopping.result(timeout=3)
                try:
                    record = request.result(timeout=3)
                    self.assertFalse(record["ok"])
                except (OSError, ValueError):
                    pass  # stop deliberately closes in-flight client sockets.
        self.assertTrue(stopped.is_set())
        self.assertFalse(service._accept_thread.is_alive())
        self.assertEqual(service._workers, set())
        self.assert_empty_storage()


if __name__ == "__main__":
    unittest.main()

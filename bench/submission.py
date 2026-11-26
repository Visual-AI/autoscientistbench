"""Receive and seal explicit submissions in a trusted host process.

The agent sends diagnostic metadata and patch bytes over an AF_UNIX socket. Only
the host determines validity, arrival time, numbering and the content digest.
The storage directory must be writable only by the host (and may be mounted read
only in the agent sandbox). Static patch validation never executes candidate code.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time

from . import patch as patchmod


_MAGIC = b"ARB1"
_HEADER = struct.Struct("!4sIQ")  # magic, metadata bytes, patch bytes
_RESPONSE_HEADER = struct.Struct("!I")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_IO_TIMEOUT_S = 5.0
_CLIENT_TIMEOUT_S = 30.0
_MAX_CONNECTIONS = 4


class _ProtocolError(ValueError):
    pass


class _Rejected(Exception):
    def __init__(self, stage, message):
        self.stage = stage
        super().__init__(message)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reject_constant(value):
    raise _ProtocolError(f"non-finite JSON value: {value}")


def _read_exact(conn, size, io_end, deadline_ts=None):
    # Preallocation prevents tiny fragments from creating millions of list items.
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        timeout = io_end - time.monotonic()
        if deadline_ts is not None:
            timeout = min(timeout, deadline_ts - time.time())
        if timeout <= 0:
            raise socket.timeout("submission receive deadline reached")
        conn.settimeout(timeout)
        count = conn.recv_into(view[offset:offset + min(size - offset, 64 * 1024)])
        if not count:
            raise _ProtocolError("incomplete submission frame")
        offset += count
    return bytes(buffer)


def _details(items):
    # Diff paths and validation messages also originate in untrusted patch bytes.
    # Bound diagnostic fields while retaining exact counts and a complete digest.
    return [str(item)[:1024] for item in items[:200]]


def client_submit(socket_path, patch_bytes, final=False, msg="", self_reported_val=None):
    """Submit a byte snapshot and return the host record; transport errors raise.

    The server independently validates all fields. An acknowledgement failure
    does not prove the submission was not stored; sealed host records are final.
    """
    if not isinstance(patch_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("patch_bytes must be bytes-like")
    patch_bytes = bytes(patch_bytes)
    metadata = _json_bytes({"final": final, "msg": msg, "self_reported_val": self_reported_val})
    if len(metadata) > _MAX_METADATA_BYTES:
        raise ValueError("submission metadata is too large")
    io_end = time.monotonic() + _CLIENT_TIMEOUT_S
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(_CLIENT_TIMEOUT_S)
        conn.connect(os.fspath(socket_path))
        send_error = None
        try:
            for data in (_HEADER.pack(_MAGIC, len(metadata), len(patch_bytes)), metadata, patch_bytes):
                timeout = io_end - time.monotonic()
                if timeout <= 0:
                    raise socket.timeout("submission communication timed out")
                conn.settimeout(timeout)
                conn.sendall(data)
        except OSError as exc:
            # Expiry/length rejection may arrive before the client finishes sending.
            send_error = exc
        try:
            header = _read_exact(conn, _RESPONSE_HEADER.size, io_end)
        except (OSError, _ProtocolError):
            if send_error is not None:
                raise send_error
            raise
        length, = _RESPONSE_HEADER.unpack(header)
        if length > _MAX_RESPONSE_BYTES:
            raise _ProtocolError("submission response is too large")
        record = json.loads(_read_exact(conn, length, io_end), parse_constant=_reject_constant)
        if not isinstance(record, dict):
            raise _ProtocolError("submission response must be a JSON object")
        return record


class SubmissionBroker:
    """A bounded host service whose stop() joins every in-flight request.

    Complete, statically invalid patches are recorded with ``ok=False``. Expired,
    malformed, oversized and interrupted requests have no numbered record.
    """

    def __init__(self, socket_path, storage_dir, deadline_ts, patch_limits):
        self.socket_path = Path(socket_path)
        self.storage_dir = Path(storage_dir)
        self.deadline_ts = float(deadline_ts)
        if not math.isfinite(self.deadline_ts):
            raise ValueError("deadline_ts must be finite")
        self.patch_limits = copy.deepcopy(patch_limits)
        self.max_bytes = int(self.patch_limits.get("max_bytes", 2 * 1024 * 1024))
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self._state_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stopping = threading.Event()
        self._workers = set()
        self._connections = set()
        self._listener = None
        self._accept_thread = None
        self._socket_identity = None
        self._started = False
        self._next_number = 1

    def start(self):
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("submission broker has been stopped")
            if self._started:
                return self
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            numbers = [int(p.stem) for p in self.storage_dir.iterdir()
                       if p.suffix in (".json", ".patch") and p.stem.isdecimal()]
            self._next_number = max(numbers, default=0) + 1
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                # Never unlink an existing endpoint belonging to another broker.
                listener.bind(os.fspath(self.socket_path))
                st = self.socket_path.lstat()
                self._socket_identity = (st.st_dev, st.st_ino)
                os.chmod(self.socket_path, 0o600)
                listener.listen(16)
                listener.settimeout(0.1)
                self._listener = listener
                self._accept_thread = threading.Thread(
                    target=self._accept_loop, name="submission-broker", daemon=True)
                self._started = True
                self._accept_thread.start()
            except BaseException:
                listener.close()
                self._unlink_socket()
                self._started = False
                raise
        return self

    def stop(self):
        """Stop admission, interrupt socket reads and join all sealing workers."""
        with self._stop_lock:
            self._stopping.set()
            with self._state_lock:
                listener = self._listener
                connections = list(self._connections)
                accept_thread = self._accept_thread
            if listener is not None:
                listener.close()
            for conn in connections:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if accept_thread is not None:
                accept_thread.join()
            # The accept thread cannot register more workers after this point.
            with self._state_lock:
                workers = list(self._workers)
            for worker in workers:
                worker.join()
            self._unlink_socket()

    def _unlink_socket(self):
        if self._socket_identity is None:
            return
        try:
            st = self.socket_path.lstat()
            if (st.st_dev, st.st_ino) == self._socket_identity:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _check_open(self):
        if time.time() > self.deadline_ts:
            raise _Rejected("deadline", "submission deadline has passed")
        if self._stopping.is_set():
            raise _Rejected("stopped", "submission service is stopping")

    @staticmethod
    def _failure(stage, message):
        return {"ok": False, "source": "submit", "stage": stage,
                "ts": time.time(), "problems": [str(message)[:1024]]}

    @staticmethod
    def _respond(conn, record):
        try:
            payload = _json_bytes(record)
            if len(payload) > _MAX_RESPONSE_BYTES:
                return
            conn.settimeout(0.2)
            conn.sendall(_RESPONSE_HEADER.pack(len(payload)) + payload)
        except (OSError, ValueError, UnicodeError):
            pass

    def _accept_loop(self):
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._state_lock:
                if self._stopping.is_set():
                    conn.close()
                    return
                if len(self._workers) < _MAX_CONNECTIONS:
                    worker = threading.Thread(target=self._serve, args=(conn,),
                                              name="submission-request", daemon=True)
                    self._workers.add(worker)
                    self._connections.add(conn)
                    try:
                        worker.start()
                    except RuntimeError:
                        self._workers.discard(worker)
                        self._connections.discard(conn)
                        conn.close()
                    continue
            self._respond(conn, self._failure("busy", "submission service is busy; retry"))
            conn.close()

    def _serve(self, conn):
        try:
            record = self._receive(conn)
        except _Rejected as exc:
            record = self._failure(exc.stage, str(exc))
        except socket.timeout:
            stage = "deadline" if time.time() > self.deadline_ts else "protocol"
            record = self._failure(stage, "submission receive deadline reached")
        except (ValueError, UnicodeError, RecursionError, OSError) as exc:
            record = self._failure("protocol", str(exc))
        except Exception:
            # A static validator/storage failure must never create a valid record.
            record = self._failure("submit", "host submission validation failed")
        try:
            self._respond(conn, record)
        finally:
            conn.close()
            with self._state_lock:
                self._connections.discard(conn)
                self._workers.discard(threading.current_thread())

    def _receive(self, conn):
        self._check_open()
        io_end = time.monotonic() + _IO_TIMEOUT_S
        header = _read_exact(conn, _HEADER.size, io_end, self.deadline_ts)
        magic, metadata_size, patch_size = _HEADER.unpack(header)
        if magic != _MAGIC:
            raise _ProtocolError("unsupported submission protocol")
        if not 2 <= metadata_size <= _MAX_METADATA_BYTES:
            raise _ProtocolError("submission metadata length is invalid")
        if patch_size > self.max_bytes:
            raise _ProtocolError(f"patch exceeds {self.max_bytes} byte limit")
        metadata = json.loads(_read_exact(conn, metadata_size, io_end, self.deadline_ts),
                              parse_constant=_reject_constant)
        if not isinstance(metadata, dict):
            raise _ProtocolError("submission metadata must be a JSON object")
        final, msg = metadata.get("final", False), metadata.get("msg", "")
        if type(final) is not bool or not isinstance(msg, str) or len(msg) > 16384:
            raise _ProtocolError("invalid submission diagnostics")
        # Only these diagnostic fields are accepted. Ignore client ts/n/ok/hash
        # and all other claims, including the purported submission source.
        diagnostics = {"final": final, "msg": msg,
                       "self_reported_val": metadata.get("self_reported_val")}
        _json_bytes(diagnostics)  # reject unsupported Unicode before any writes
        patch_bytes = _read_exact(conn, patch_size, io_end, self.deadline_ts)
        received_ts = time.time()
        self._check_open()
        ok, problems, files = patchmod.check_patch(
            patch_bytes, self.patch_limits,
            protected_paths=self.patch_limits.get("protected_paths", ()))
        validated_ts = time.time()
        self._check_open()
        record = {**diagnostics, "source": "submit", "ok": bool(ok),
                  "received_ts": received_ts, "validated_ts": validated_ts,
                  "problems": _details(problems),
                  "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
                  "bytes": len(patch_bytes), "files": len(files),
                  "file_list": _details([f["path"] for f in files])}
        return self._seal(patch_bytes, record)

    def _write_pending(self, content):
        fd, name = tempfile.mkstemp(prefix=".pending-", dir=self.storage_dir)
        path = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def _seal(self, patch_bytes, record):
        with self._commit_lock:
            self._check_open()
            number = self._next_number
            final_patch = self.storage_dir / f"{number:03d}.patch"
            final_record = self.storage_dir / f"{number:03d}.json"
            pending, published = [], []
            try:
                pending.append(self._write_pending(patch_bytes))
                self._check_open()
                record.update(n=number, ts=time.time())
                pending.append(self._write_pending(_json_bytes(record) + b"\n"))
                self._check_open()
                # Hard links publish complete files without overwriting anything.
                os.link(pending[0], final_patch)
                published.append(final_patch)
                os.link(pending[1], final_record)
                published.append(final_record)
                for path in pending:
                    path.unlink()
                directory_fd = os.open(self.storage_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self._check_open()
                self._next_number += 1
                return record
            except BaseException:
                for path in published:
                    path.unlink(missing_ok=True)
                raise
            finally:
                for path in pending:
                    path.unlink(missing_ok=True)

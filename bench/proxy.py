#!/usr/bin/env python3
"""Host side of the sandbox network: an HTTP proxy on a unix socket with two jobs.

1. CONNECT <host>:<port>   -> only hosts on the allowlist (PyPI / PyTorch index / the harness's own API);
                              bytes are tunnelled untouched (TLS end-to-end), the host and byte counts are logged.
2. /t/<run>/<stage>/v1/... -> the model route for harnesses that speak an OpenAI-style API (codex): forwarded to the
                              configured upstream with the API key injected here (the key never enters the sandbox),
                              streamed back, and every exchange recorded under trace_dir/<run>/<stage>/req_NNNN.json
                              (same {request, sse_events} idea as pipeline/codex_tools/trace_proxy.py).
Everything else is 403. Plain-HTTP absolute-URI requests are forwarded only to allowlisted hosts.
"""
import argparse
import http.client
import json
import os
import re
import select
import socket
import ssl
import threading
import time
import urllib.parse
from pathlib import Path

if __package__:
    from .connect_relay import connect_socket
else:
    from connect_relay import connect_socket

ROUTE = re.compile(r"^/t/([^/]+)/([^/]+)(/v1(?:/.*)?)$")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "proxy-connection", "host", "content-length"}
IMAGE_NOTE = "[image omitted: this model endpoint takes no images; print or describe the data instead]"


def strip_images(req):
    """Replace image parts of a Responses request with a text note; returns how many. codex offers view_image and config
    cannot remove it, and a vLLM started with an image limit of 0 answers 400 on every retry once a history holds an image."""
    n = 0

    def fix(parts):
        nonlocal n
        out = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") in ("input_image", "output_image", "image_url", "image"):
                out.append({"type": "input_text", "text": IMAGE_NOTE})
                n += 1
            else:
                out.append(p)
        return out

    items = req.get("input") if isinstance(req, dict) else None
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        if isinstance(it.get("content"), list):
            it["content"] = fix(it["content"])
        if it.get("type") == "function_call_output" and isinstance(it.get("output"), list):
            it["output"] = fix(it["output"])
            if all(isinstance(p, dict) and p.get("type") == "input_text" for p in it["output"]):
                it["output"] = "\n".join(p.get("text", "") for p in it["output"])
    return n


class ProxyConfig:
    def __init__(self, allow_hosts=(), model_upstream=None, trace_dir=None, log_path=None, upstream_timeout=1800.0,
                 connect_relay=None):
        self.allow = [h.lower() for h in allow_hosts]
        self.model_upstream = model_upstream or None   # {"base_url", "api_key", "headers": {...}}
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.log_path = Path(log_path) if log_path else None
        self.upstream_timeout = upstream_timeout
        relay_path = os.environ.get('ARB_CONNECT_RELAY_CONFIG')
        self.connect_relay = connect_relay if connect_relay is not None else (
            json.loads(Path(relay_path).read_text()) if relay_path else None)

    def host_allowed(self, host):
        h = host.lower().rstrip(".")
        for a in self.allow:
            if a.startswith("*.") and (h.endswith(a[1:]) or h == a[2:]):
                return True
            if h == a:
                return True
        return False


class Proxy:
    def __init__(self, cfg: ProxyConfig, unix_path=None, tcp=None):
        self.cfg = cfg
        self.unix_path = unix_path
        self.tcp = tcp
        self.lock = threading.Lock()
        self.seq = {}
        self.stats = {"connect_ok": 0, "connect_denied": 0, "model_calls": 0, "model_errors": 0, "in_tokens": 0, "out_tokens": 0}
        self.socks = []
        self.stopping = threading.Event()
        self.threads = []

    # ---- lifecycle -----------------------------------------------------------
    def start(self):
        if self.unix_path:
            p = Path(self.unix_path)
            if p.exists():
                p.unlink()
            p.parent.mkdir(parents=True, exist_ok=True)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(str(p))
            os.chmod(str(p), 0o666)
            s.listen(256)
            self.socks.append(s)
        if self.tcp:
            host, port = self.tcp.rsplit(":", 1)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            s.listen(256)
            self.socks.append(s)
        for s in self.socks:
            t = threading.Thread(target=self._accept_loop, args=(s,), daemon=True)
            t.start()
            self.threads.append(t)
        return self

    def stop(self):
        self.stopping.set()
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass
        if self.unix_path and os.path.exists(self.unix_path):
            try:
                os.unlink(self.unix_path)
            except OSError:
                pass

    def _accept_loop(self, srv):
        while not self.stopping.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    # ---- helpers -------------------------------------------------------------
    def log(self, rec):
        rec = dict(rec, ts=round(time.time(), 3))
        if self.cfg.log_path:
            with self.lock:
                with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _reply(c, status, reason, body=b"", ctype="text/plain"):
        try:
            c.sendall((f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                       f"Connection: close\r\n\r\n").encode() + body)
        except OSError:
            pass

    @staticmethod
    def _read_head(c, limit=256 * 1024):
        buf = b""
        c.settimeout(60)
        while b"\r\n\r\n" not in buf:
            d = c.recv(65536)
            if not d:
                return None, b""
            buf += d
            if len(buf) > limit:
                return None, b""
        head, rest = buf.split(b"\r\n\r\n", 1)
        return head.decode("latin-1"), rest

    @staticmethod
    def _parse_head(head):
        lines = head.split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) < 2:
            return None, None, None, {}
        method, target = parts[0], parts[1]
        version = parts[2] if len(parts) > 2 else "HTTP/1.0"
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        return method, target, version, headers

    @staticmethod
    def _read_body(c, rest, headers):
        n = int(headers.get("content-length") or 0)
        body = rest
        while len(body) < n:
            d = c.recv(min(65536, n - len(body)))
            if not d:
                break
            body += d
        return body[:n]

    # ---- dispatcher ------------------------------------------------------------
    def _handle(self, c):
        t0 = time.time()
        try:
            head, rest = self._read_head(c)
            if head is None:
                c.close()
                return
            method, target, version, headers = self._parse_head(head)
            if method is None:
                self._reply(c, 400, "Bad Request", b"bad request line\n")
                c.close()
                return
            if method == "CONNECT":
                return self._connect(c, target, t0)
            if target.startswith("http://") or target.startswith("https://"):
                return self._absolute(c, method, target, headers, rest, t0)
            if target in ("/healthz", "/"):
                with self.lock:
                    st = dict(self.stats)
                return self._reply(c, 200, "OK", (json.dumps(st) + "\n").encode(), "application/json")
            m = ROUTE.match(target)
            if m and self.cfg.model_upstream:
                body = self._read_body(c, rest, headers)
                return self._model(c, method, m.group(1), m.group(2), m.group(3), headers, body, t0)
            self.log({"kind": "deny", "method": method, "target": target[:200]})
            self._reply(c, 403, "Forbidden", b"not on the allowlist\n")
        except Exception as e:  # noqa: BLE001
            self.log({"kind": "error", "error": f"{type(e).__name__}: {e}"})
            try:
                self._reply(c, 502, "Bad Gateway", f"{type(e).__name__}: {e}\n".encode())
            except Exception:
                pass
        finally:
            try:
                c.close()
            except OSError:
                pass

    # ---- CONNECT tunnel --------------------------------------------------------
    def _connect(self, c, target, t0):
        host, _, port = target.rpartition(":")
        port = int(port or 443)
        if not host or not self.cfg.host_allowed(host) or port not in (443, 80):
            with self.lock:
                self.stats["connect_denied"] += 1
            self.log({"kind": "connect", "host": host, "port": port, "allowed": False})
            self._reply(c, 403, "Forbidden", f"host {host}:{port} is not on the allowlist\n".encode())
            return
        try:
            u = connect_socket(host, port, timeout=30, relay=self.cfg.connect_relay)
        except OSError as e:
            self.log({"kind": "connect", "host": host, "port": port, "allowed": True, "error": str(e)})
            self._reply(c, 502, "Bad Gateway", f"upstream connect failed: {e}\n".encode())
            return
        with self.lock:
            self.stats["connect_ok"] += 1
        c.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        c.settimeout(None)
        up = down = 0
        try:
            while True:
                r, _, _ = select.select([c, u], [], [], 600)
                if not r:
                    break
                done = False
                for s in r:
                    d = s.recv(65536)
                    if not d:
                        done = True
                        break
                    if s is c:
                        u.sendall(d)
                        up += len(d)
                    else:
                        c.sendall(d)
                        down += len(d)
                if done:
                    break
        except OSError:
            pass
        finally:
            try:
                u.close()
            except OSError:
                pass
        self.log({"kind": "connect", "host": host, "port": port, "allowed": True, "up": up, "down": down,
                  "dur": round(time.time() - t0, 2)})

    # ---- plain absolute-URI forwarding (allowlisted hosts only) -----------------
    def _absolute(self, c, method, target, headers, rest, t0):
        u = urllib.parse.urlsplit(target)
        host = u.hostname or ""
        if not self.cfg.host_allowed(host):
            self.log({"kind": "deny", "method": method, "target": target[:200]})
            self._reply(c, 403, "Forbidden", b"host not on the allowlist\n")
            return
        body = self._read_body(c, rest, headers)
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(
            host, u.port, timeout=self.cfg.upstream_timeout)
        fwd = {k: v for k, v in headers.items() if k not in HOP}
        conn.request(method, path, body=body or None, headers=fwd)
        self._relay_response(c, conn.getresponse())
        conn.close()
        self.log({"kind": "http", "host": host, "method": method, "dur": round(time.time() - t0, 2)})

    # ---- model route --------------------------------------------------------------
    def _model(self, c, method, run, stage, rest, headers, body, t0):
        up = self.cfg.model_upstream
        base = urllib.parse.urlsplit(up["base_url"])
        upath = base.path.rstrip("/") + rest[len("/v1"):]
        conn_cls = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(base.hostname, base.port, timeout=self.cfg.upstream_timeout)
        fwd = {k: v for k, v in headers.items() if k not in HOP and k not in ("authorization",)}
        if up.get("api_key"):
            fwd["Authorization"] = "Bearer " + up["api_key"]
        elif headers.get("authorization"):
            fwd["Authorization"] = headers["authorization"]
        for k, v in (up.get("headers") or {}).items():
            fwd[k.lower()] = v
        # Incoming header names are already lowercase. A second title-case
        # Content-Type produces duplicate headers, rejected by strict APIs.
        fwd.setdefault("content-type", "application/json")
        req_json = None
        try:
            req_json = json.loads(body.decode("utf-8")) if body else None
        except ValueError:
            req_json = None
        if up.get("strip_images") and req_json is not None and strip_images(req_json):
            body = json.dumps(req_json, ensure_ascii=False).encode("utf-8")
        with self.lock:
            self.stats["model_calls"] += 1
        status, chunks = None, []
        try:
            # API-mode calls also need the operator relay on isolated workers.
            # Keep HTTPS certificate verification and TLS end to end; the
            # relay sees neither the provider key nor model request contents.
            if (base.scheme == "https" and self.cfg.connect_relay
                    and base.hostname.lower().rstrip('.') in self.cfg.connect_relay.get('hosts', [])):
                raw = connect_socket(base.hostname, base.port or 443,
                                     timeout=self.cfg.upstream_timeout, relay=self.cfg.connect_relay)
                try:
                    conn.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=base.hostname)
                except BaseException:
                    raw.close()
                    raise
            conn.request(method, upath, body=body or None, headers=fwd)
            resp = conn.getresponse()
            status = resp.status
            self._relay_response(c, resp, sink=chunks)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.stats["model_errors"] += 1
            self.log({"kind": "model", "run": run, "stage": stage, "path": rest, "error": f"{type(e).__name__}: {e}"})
            if status is None:
                self._reply(c, 502, "Bad Gateway", f"upstream error: {type(e).__name__}: {e}\n".encode())
            return
        finally:
            conn.close()
        text = b"".join(chunks).decode("utf-8", "replace")
        usage = _usage_from_response(text)
        with self.lock:
            self.stats["in_tokens"] += usage.get("input_tokens", 0)
            self.stats["out_tokens"] += usage.get("output_tokens", 0)
        self._record(run, stage, {"t0": t0, "elapsed_s": round(time.time() - t0, 2), "path": rest, "status": status,
                                  "request": req_json if req_json is not None else body.decode("utf-8", "replace")[:200000],
                                  "response_text": text, "usage": usage})
        self.log({"kind": "model", "run": run, "stage": stage, "path": rest, "status": status,
                  "dur": round(time.time() - t0, 2), "bytes": len(text), **{k: v for k, v in usage.items() if v}})

    def _relay_response(self, c, resp, sink=None):
        hdrs = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP]
        head = f"HTTP/1.1 {resp.status} {resp.reason}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs)
        head += "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        c.sendall(head.encode("latin-1"))
        while True:
            d = resp.read1(65536)
            if not d:
                break
            if sink is not None:
                sink.append(d)
            c.sendall(b"%x\r\n" % len(d) + d + b"\r\n")
        c.sendall(b"0\r\n\r\n")

    def _record(self, run, stage, rec):
        if not self.cfg.trace_dir:
            return
        d = self.cfg.trace_dir / _safe(run) / _safe(stage)
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            key = (run, stage)
            if key not in self.seq:
                # A resumed worker creates a new Proxy. Continue the archive
                # numbering instead of overwriting the preceding attempt.
                self.seq[key] = max((int(p.stem[4:]) for p in d.glob('req_*.json')
                                     if p.stem[4:].isdigit()), default=0)
            n = self.seq[key] + 1
            self.seq[key] = n
        with open(d / f"req_{n:04d}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120] or "x"


def _usage_from_response(text):
    """Pull token usage out of an SSE stream (response.completed) or a plain JSON body."""
    usage = {}
    try:
        if text.lstrip().startswith("{"):
            j = json.loads(text)
            u = j.get("usage") or (j.get("response") or {}).get("usage") or {}
            usage = {k: int(v) for k, v in u.items() if isinstance(v, (int, float))}
        else:
            for line in text.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        j = json.loads(payload)
                    except ValueError:
                        continue
                    u = (j.get("response") or {}).get("usage") or j.get("usage")
                    if u:
                        usage = {k: int(v) for k, v in u.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return usage


def main():
    ap = argparse.ArgumentParser(description="allowlist proxy on a unix socket (debug/standalone)")
    ap.add_argument("--unix")
    ap.add_argument("--tcp", help="also listen on host:port")
    ap.add_argument("--allow", default="", help="comma-separated hosts (supports *.suffix)")
    ap.add_argument("--upstream", help="model upstream base_url, e.g. http://127.0.0.1:18899/v1")
    ap.add_argument("--api-key-env")
    ap.add_argument("--trace-dir")
    ap.add_argument("--log")
    a = ap.parse_args()
    up = None
    if a.upstream:
        up = {"base_url": a.upstream, "api_key": os.environ.get(a.api_key_env) if a.api_key_env else None}
    cfg = ProxyConfig([h for h in a.allow.split(",") if h], up, a.trace_dir, a.log)
    p = Proxy(cfg, a.unix, a.tcp).start()
    print(f"proxy listening unix={a.unix} tcp={a.tcp}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        p.stop()


if __name__ == "__main__":
    main()

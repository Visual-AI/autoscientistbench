#!/usr/bin/env python3
"""Inside the sandbox: forward 127.0.0.1:<port> to a unix socket bound in from the host (the allowlist proxy).
stdlib only, threads, no buffering tricks. The sandbox has no other way to the network."""
import argparse
import os
import socket
import sys
import threading


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        try:
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(c, unix_path):
    try:
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        u.connect(unix_path)
    except OSError as e:
        try:
            c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass
        c.close()
        return
    t = threading.Thread(target=pipe, args=(u, c), daemon=True)
    t.start()
    pipe(c, u)
    t.join(timeout=30)
    for s in (c, u):
        try:
            s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8080")
    ap.add_argument("--unix", required=True)
    ap.add_argument("--ready", default=None)
    a = ap.parse_args()
    host, port = a.listen.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(128)
    if a.ready:
        with open(a.ready, "w") as f:
            f.write(str(os.getpid()))
    while True:
        try:
            c, _ = srv.accept()
        except OSError:
            break
        threading.Thread(target=handle, args=(c, a.unix), daemon=True).start()


if __name__ == "__main__":
    main()

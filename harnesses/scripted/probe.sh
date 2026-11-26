#!/usr/bin/env bash
# Closure probes: everything here must FAIL except the allowlisted index and the bench commands.
set +e
echo "== whoami: $(id -u) cwd: $(pwd) nproc: $(nproc)"
echo "== ls /mnt";        ls /mnt 2>&1 | head -2
echo "== ls /";           ls / | tr '\n' ' '; echo
echo "== judge visible?"; ls /ws/judge /ws/../judge 2>&1 | head -2
echo "== direct DNS";     getent hosts example.com 2>&1 | head -1 || echo "no dns (expected)"
echo "== curl example.com via proxy (expect CONNECT 403)"; curl -s -m 20 -o /dev/null -w 'connect=%{http_connect} http=%{http_code}\n' https://example.com || true
echo "== curl pypi via proxy (expect CONNECT 200, http 200)"; curl -s -m 60 -o /dev/null -w 'connect=%{http_connect} http=%{http_code}\n' https://pypi.org/simple/pip/
echo "== bench status"; bench status
echo "== python"; python -c 'import sys, numpy; print(sys.executable, numpy.__version__)'
echo "== probes done"

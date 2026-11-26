"""Candidate-side legal oracle channel; holds no private state."""
import os, socket, threading
from private_wire import recv_frame, send_frame
_socket = None
_pid = None
_lock = threading.Lock()

def call(method, *args, **kwargs):
    global _socket, _pid
    with _lock:
        if _socket is None or _pid != os.getpid():
            if _socket is not None: _socket.close()
            _socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _socket.connect(os.environ['BENCH_RPC_SOCKET'])
            _pid = os.getpid()
        send_frame(_socket, {'method': method, 'args': list(args), 'kwargs': kwargs})
        reply = recv_frame(_socket)
        if not isinstance(reply, dict) or reply.get('ok') is not True:
            raise RuntimeError('private oracle rejected request')
        return reply['result']

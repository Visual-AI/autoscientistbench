"""Optional operator-side HTTPS relay; payloads remain opaque TLS bytes."""
import base64
import socket


def connect_socket(host, port, *, timeout=30, relay=None):
    if not relay or host.lower().rstrip('.') not in relay.get('hosts', []):
        return socket.create_connection((host, port), timeout=timeout)
    if port != 443:
        raise OSError('Configured relay only accepts HTTPS')
    address, relay_port = relay['public_host'], int(relay['public_port'])
    sock = socket.create_connection((address, relay_port), timeout=timeout)
    try:
        authority = f'[{address}]:{relay_port}' if ':' in address else f'{address}:{relay_port}'
        auth = base64.b64encode(('relay:' + relay['password']).encode()).decode()
        # Platform port routing requires Host on the outer HTTP request.
        # HTTP/1.0 CONNECT without Host is routed to the gateway's default service.
        sock.sendall((f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {authority}\r\n'
                      f'Proxy-Authorization: Basic {auth}\r\n\r\n').encode('ascii'))
        head = bytearray()
        while not head.endswith(b'\r\n\r\n'):
            part = sock.recv(1)
            if not part or len(head) >= 16384:
                raise OSError('Incomplete relay handshake')
            head.extend(part)
        fields = bytes(head).split(b'\r\n', 1)[0].split()
        if len(fields) < 2 or fields[1] != b'200':
            status = fields[1].decode('ascii', 'replace') if len(fields) > 1 else 'invalid'
            raise OSError('Relay CONNECT returned HTTP ' + status)
        return sock
    except Exception:
        sock.close()
        raise

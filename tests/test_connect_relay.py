import base64
import socket
import threading
import unittest
from unittest.mock import patch

from bench.connect_relay import connect_socket


class ConnectRelayTests(unittest.TestCase):
    def test_authenticated_http11_preserves_tunnel_bytes(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        captured = []
        def serve():
            conn, _ = listener.accept()
            with conn:
                head = b''
                while not head.endswith(b'\r\n\r\n'):
                    head += conn.recv(1)
                captured.append(head)
                conn.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\nopaque-tls-data')
            listener.close()
        thread = threading.Thread(target=serve)
        thread.start()
        port = listener.getsockname()[1]
        relay = {'public_host': '127.0.0.1', 'public_port': port,
                 'password': 'test-password', 'hosts': ['api.kimi.com']}
        with connect_socket('api.kimi.com', 443, relay=relay) as conn:
            self.assertEqual(conn.recv(100), b'opaque-tls-data')
        thread.join(2)
        self.assertIn(b'CONNECT api.kimi.com:443 HTTP/1.1\r\n', captured[0])
        self.assertIn(f'Host: 127.0.0.1:{port}\r\n'.encode(), captured[0])
        self.assertIn(base64.b64encode(b'relay:test-password'), captured[0])

    def test_unlisted_hosts_keep_existing_direct_route(self):
        with patch('bench.connect_relay.socket.create_connection') as direct:
            connect_socket('pypi.org', 443, relay={'hosts': ['api.kimi.com']})
        direct.assert_called_once_with(('pypi.org', 443), timeout=30)

    def test_relay_failure_is_not_retried_directly(self):
        relay = {'public_host': 'relay.example', 'public_port': 123,
                 'password': 'secret', 'hosts': ['api.kimi.com']}
        with patch('bench.connect_relay.socket.create_connection') as dial:
            stream = dial.return_value
            stream.recv.side_effect = [bytes([x]) for x in b'HTTP/1.1 407 Denied\r\n\r\n']
            with self.assertRaisesRegex(OSError, '407'):
                connect_socket('api.kimi.com', 443, relay=relay)
            self.assertEqual(dial.call_count, 1)
            stream.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

"""Bounded JSON/numeric-array RPC; no pickle or object arrays."""
import base64, json, math, struct
MAX_MESSAGE = 128 * 1024 * 1024

def encode(value):
    if type(value).__module__.split('.')[0] == 'numpy':
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'biufc' or value.nbytes > MAX_MESSAGE // 2:
                raise ValueError('unsupported RPC array')
            a = np.ascontiguousarray(value)
            return {'__numeric_array_v1__': True, 'dtype': a.dtype.str,
                    'shape': list(a.shape), 'data': base64.b64encode(a.tobytes()).decode('ascii')}
        return value.item()
    raise TypeError('unsupported RPC value')

def decode(value):
    if value.get('__numeric_array_v1__') is True:
        import numpy as np
        if set(value) != {'__numeric_array_v1__', 'dtype', 'shape', 'data'}:
            raise ValueError('malformed RPC array')
        dtype, shape = np.dtype(value['dtype']), value['shape']
        if dtype.kind not in 'biufc' or not isinstance(shape, list) or len(shape) > 8:
            raise ValueError('unsupported RPC array')
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid RPC shape')
        size = math.prod(shape) * dtype.itemsize
        if size > MAX_MESSAGE // 2: raise ValueError('RPC array too large')
        data = base64.b64decode(value['data'], validate=True)
        if len(data) != size: raise ValueError('RPC array size mismatch')
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return value

def send_frame(sock, value):
    data = json.dumps(value, default=encode, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_MESSAGE: raise ValueError('RPC message too large')
    sock.sendall(struct.pack('!I', len(data)) + data)

def exact(sock, n):
    chunks, left = [], n
    while left:
        part = sock.recv(min(left, 1024 * 1024))
        if not part: raise EOFError('RPC closed')
        chunks.append(part); left -= len(part)
    return b''.join(chunks)

def recv_frame(sock):
    size = struct.unpack('!I', exact(sock, 4))[0]
    if not 0 < size <= MAX_MESSAGE: raise ValueError('RPC message too large')
    def bad(_): raise ValueError('nonfinite RPC JSON')
    return json.loads(exact(sock, size), object_hook=decode, parse_constant=bad)

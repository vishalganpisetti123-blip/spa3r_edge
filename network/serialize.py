import io
import struct

import numpy as np


def serialize_tensor(numpy_array: np.ndarray) -> bytes:
    """Serialize a NumPy array into a byte stream with a length header."""
    buffer = io.BytesIO()
    np.save(buffer, numpy_array, allow_pickle=False)
    payload = buffer.getvalue()
    header = struct.pack(">I", len(payload))
    return header + payload


def deserialize_tensor(connection) -> np.ndarray:
    """Read a length-prefixed NumPy array from a socket connection."""
    raw_header = recvall(connection, 4)
    if not raw_header:
        return None
    msg_len = struct.unpack(">I", raw_header)[0]
    payload = recvall(connection, msg_len)
    if not payload:
        return None
    buffer = io.BytesIO(payload)
    return np.load(buffer, allow_pickle=False)


def recvall(sock, n: int) -> bytes:
    """Receive exactly n bytes from a socket, or return None on EOF."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

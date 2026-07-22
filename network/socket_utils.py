import socket


def create_client_socket(host: str, port: int) -> socket.socket:
    """Initialize and connect a standard TCP client socket."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket
¸
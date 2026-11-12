"""The shared offline guard in tests/conftest.py. Nothing here leaves the machine."""
import socket

import pytest
from conftest import NetworkBlocked


def test_remote_connection_is_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkBlocked, match="connect"):
        s.connect(("93.184.216.34", 80))


def test_connect_ex_is_blocked_too():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkBlocked):
        s.connect_ex(("93.184.216.34", 80))


def test_remote_name_lookup_is_blocked():
    with pytest.raises(NetworkBlocked, match="resolve"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_requests_to_an_api_fail_fast():
    requests = pytest.importorskip("requests")
    with pytest.raises((NetworkBlocked, requests.exceptions.ConnectionError)):
        requests.get("https://openai.rc.asu.edu/v1/models", timeout=1)


def test_loopback_still_works():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            conn, _ = server.accept()
            conn.close()
    assert socket.getaddrinfo("localhost", 80)

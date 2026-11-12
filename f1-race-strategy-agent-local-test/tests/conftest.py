"""Shared pytest setup: every collected test runs offline.

A test that resolves a remote host name or opens a connection to a remote address fails
with NetworkBlocked, instead of quietly depending on the internet (FastF1, an LLM API).
Loopback and Unix sockets stay allowed, so worker pools and local servers keep working.
Race data comes from the committed cache/ in offline mode.
"""
import socket

import pytest

_LOCAL_NAMES = {None, "", "localhost", "::1", "0.0.0.0"}


class NetworkBlocked(AssertionError):
    """A test tried to reach the network."""


def _local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    return host in _LOCAL_NAMES or (isinstance(host, str) and host.startswith("127."))


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    connect, connect_ex, getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def check(sock, address):
        if sock.family != getattr(socket, "AF_UNIX", None) and not _local(address[0]):
            raise NetworkBlocked(f"test tried to connect to {address!r}; tests run offline")

    def guarded_connect(sock, address):
        check(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(sock, address)
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _local(host):
            raise NetworkBlocked(f"test tried to resolve {host!r}; tests run offline")
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

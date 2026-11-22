"""Shared pytest setup: every collected test runs offline and logs nothing to wandb.

A test that resolves a remote host name or opens a connection to a remote address fails
with NetworkBlocked, instead of quietly depending on the internet (FastF1, an LLM API).
Loopback and Unix sockets stay allowed, so worker pools and local servers keep working.
Race data comes from the committed cache/ in offline mode.
"""
import os
import socket

import pytest

os.environ["WANDB_MODE"] = "disabled"   # set at import too, before any worker process starts (compute.training_run reads it)

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


def _drop_cached_belief():
    """agent_tools.belief_config() reads belief.yaml once per process; a copy cached under the
    other default (at collection, by a module-level BELIEF) would carry it into the test."""
    import sys
    tools = sys.modules.get("src.sim.agent_tools")
    if tools is not None:
        tools._BELIEF.pop("config", None)


def _set_proxy_default(enabled):
    from src.sim.config import CompoundProxy
    field = CompoundProxy.model_fields["enabled"]
    if field.default != enabled:
        field.default = enabled
        CompoundProxy.model_rebuild(force=True)
        _drop_cached_belief()


def _set_cliff_default(enabled):
    from src.sim.config import WearCliff
    field = WearCliff.model_fields["enabled"]
    if field.default != enabled:
        field.default = enabled
        WearCliff.model_rebuild(force=True)
        _drop_cached_belief()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """`pytest.mark.compound_proxy_off` (a module's pytestmark, or one test's): SimConfig's
    default compound_proxy.enabled is False while the test runs, its fixtures included (set
    here, before any fixture of the test is built), so a config built without naming it is the
    simulator as it was before proxying (src/sim/proxy.py), bit for bit. The modules that carry
    it test other mechanics on synthetic two-compound races (SOFT and MEDIUM, no HARD) and pin
    those compounds' fits, reports and limits; with the proxy on, every such race would also
    price HARD by the cross-race step. The proxy itself is tested with its default on
    (tests/test_sim_proxy.py), and the unmarked tests run on the default. A config that names
    compound_proxy keeps what it names. In-process only: a spawned worker sees the real
    default."""
    _set_proxy_default(item.get_closest_marker("compound_proxy_off") is None)
    # `pytest.mark.wear_cliff_off`: likewise SimConfig's default wear_cliff.enabled is False, for
    # tests that pin a fit's exactness on synthetic races whose real stints the cliff would price
    # (sim/cliff.py); the cliff itself is tested with its default on
    _set_cliff_default(item.get_closest_marker("wear_cliff_off") is None)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    _set_proxy_default(True)
    _set_cliff_default(True)


@pytest.fixture(autouse=True, scope="session")
def _wandb_disabled():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        yield

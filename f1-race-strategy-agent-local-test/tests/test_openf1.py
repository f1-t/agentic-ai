"""OpenF1 reads from the committed cache without touching the network."""

import pytest
import requests

from src.data_sources import openf1


def _no_network(*args, **kwargs):
    raise AssertionError("openf1 tried the network for a cached endpoint")


def test_cached_endpoint_loads_from_disk(monkeypatch):
    monkeypatch.setattr(requests, "get", _no_network)
    drivers = openf1.fetch_openf1("drivers", 7953)
    assert isinstance(drivers, list) and drivers
    assert {"driver_number", "name_acronym"} <= set(drivers[0])


def test_uncached_endpoint_would_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(openf1, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(requests, "get", _no_network)
    with pytest.raises(AssertionError, match="tried the network"):
        openf1.fetch_openf1("drivers", 7953)

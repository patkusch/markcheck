"""markcheck must work with the network switched off."""
import json
import socket

import pytest

from markcheck.cli import main
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("markcheck tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_all_fixtures_offline(no_network, capsys):
    code = main([str(FIXTURES), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 1
    assert doc["files"]
    for f in doc["files"]:
        for fd in f["findings"]:
            assert fd["status"] != "error", (f["path"], fd["summary"])
    code = main([str(FIXTURES)])
    assert code == 1

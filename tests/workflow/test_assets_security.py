"""Trusted artifact gates and explicitly mocked network failure budgets."""
import pickle
from pathlib import Path
import sys
import types
import pytest
from aurora_workflow.assets import convert_official_static, verify_source_references


def test_static_checksum_checked_before_deserialization(tmp_path):
    pytest.importorskip("numpy")
    marker = tmp_path / "must-not-exist"
    class Malicious:
        def __reduce__(self):
            return (Path.write_text, (marker, "unsafe"))
    source = tmp_path / "untrusted.pickle"
    source.write_bytes(pickle.dumps(Malicious()))
    with pytest.raises(ValueError, match="checksum"):
        convert_official_static(source, tmp_path/"out.npz")
    assert not marker.exists()


def test_source_reference_uses_fork_ancestry():
    result = verify_source_references()
    assert result["state"] == "succeeded"
    assert result["fork_source"] == "88f652f04aa75b65e410b8d00cf4ea07f2edd945"


def test_mocked_ads_retries_preserve_request_budget(tmp_path, monkeypatch):
    from aurora_workflow.data import acquire
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["url"] == "https://ads.atmosphere.copernicus.eu/api"
        def retrieve(self, product, request):
            calls.append(request)
            raise ConnectionError("explicit network failure fixture")
    monkeypatch.setitem(sys.modules,"cdsapi",types.SimpleNamespace(Client=Client))
    monkeypatch.setattr("aurora_workflow.data.time.sleep",lambda _:None)
    spec={"start_date":"2024-01-02","end_date":"2024-01-02","lead_hours":[12]}
    limits={"max_requests":3,"max_download_bytes":10_000_000_000,"retries":1}
    with pytest.raises(RuntimeError,match="ADS request failed"):
        acquire(spec,tmp_path,limits)
    assert len(calls)==2
    with pytest.raises(ValueError,match="submission count exhausted"):
        acquire(spec,tmp_path,limits)
    assert len(calls)==3

import httpx
from lib import llm_client as m

def test_list_local_models(monkeypatch):
    class R:
        status_code = 200
        def json(self): return {"models": [{"name": "llama3.2"}, {"name": "qwen3"}]}
    monkeypatch.setattr(m.httpx, "get", lambda url, timeout=5: R())
    assert m.list_local_models("http://x:11434") == ["llama3.2", "qwen3"]

def test_list_local_models_down(monkeypatch):
    def boom(url, timeout=5): raise httpx.ConnectError("no")
    monkeypatch.setattr(m.httpx, "get", boom)
    assert m.list_local_models() == []

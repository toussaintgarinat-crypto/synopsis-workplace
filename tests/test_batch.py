from lib.batch import summarize_many


def test_continues_after_error():
    def fake(url):
        if "bad" in url:
            raise ValueError("boom")
        return {"titre": url}
    out = summarize_many(["a", "bad", "c"], fake)
    assert [o["ok"] for o in out] == [True, False, True]
    assert out[1]["erreur"] == "boom"

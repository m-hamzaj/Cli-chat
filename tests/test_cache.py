import cache


def test_normalize_collapses_whitespace_and_case():
    assert cache.normalize("  Hello   World  ") == "hello world"
    assert cache.normalize("hello world") == cache.normalize("  Hello   World  ")


def test_make_key_same_for_equivalent_questions():
    k1 = cache.make_key("model-x", "What is 2+2?")
    k2 = cache.make_key("model-x", "  what is 2+2?  ")
    assert k1 == k2


def test_make_key_differs_by_model_or_text():
    base = cache.make_key("model-x", "hello")
    assert base != cache.make_key("model-y", "hello")
    assert base != cache.make_key("model-x", "goodbye")


def test_response_cache_roundtrip_and_persistence(tmp_path, monkeypatch):
    cache_file = str(tmp_path / "response_cache.json")
    monkeypatch.setattr(cache, "CACHE_PATH", cache_file)

    store = cache.ResponseCache()
    key = cache.make_key("model-x", "What's the capital of France?")
    assert store.get(key) is None

    store.put(key, "Paris")
    assert store.get(key) == "Paris"

    # a fresh instance should load what was persisted to disk
    reloaded = cache.ResponseCache()
    assert reloaded.get(key) == "Paris"

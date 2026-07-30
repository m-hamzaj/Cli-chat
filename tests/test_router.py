from router import choose_model, SMALL_MODEL, BIG_MODEL


def test_simple_mode_always_uses_big_model():
    assert choose_model("hi", "simple") == BIG_MODEL
    assert choose_model("explain in detail how TCP works", "simple") == BIG_MODEL


def test_optimized_mode_routes_short_plain_message_to_small_model():
    assert choose_model("hi there", "optimized") == SMALL_MODEL
    assert choose_model("what's 2 + 2?", "optimized") == SMALL_MODEL


def test_optimized_mode_routes_complex_keyword_to_big_model():
    assert choose_model("Explain in detail the tradeoffs of REST vs GraphQL", "optimized") == BIG_MODEL
    assert choose_model("Please compare these two approaches", "optimized") == BIG_MODEL


def test_optimized_mode_routes_long_message_to_big_model():
    long_message = "tell me about your day " * 15  # > 200 chars, no keyword match
    assert choose_model(long_message, "optimized") == BIG_MODEL

import cost_tracker


def test_price_call_known_model():
    # llama-3.1-8b-instant: $0.05 in / $0.08 out per 1M tokens
    cost = cost_tracker.price_call("llama-3.1-8b-instant", 1_000_000, 1_000_000)
    assert round(cost, 6) == round(0.05 + 0.08, 6)


def test_price_call_unknown_model_does_not_raise():
    assert cost_tracker.price_call("some-future-model", 1000, 1000) == 0.0


def test_cost_tracker_turn_and_session_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_tracker, "LOG_PATH", str(tmp_path / "costs.jsonl"))
    monkeypatch.setattr(cost_tracker, "SUMMARY_PATH", str(tmp_path / "summary.json"))

    tracker = cost_tracker.CostTracker("test-session", "optimized")
    tracker.record_call("llama-3.1-8b-instant", 1_000_000, 1_000_000, "response", turn_index=1)
    turn_cost = tracker.close_turn(1)

    assert round(turn_cost, 6) == round(0.13, 6)
    assert round(tracker.session_total(), 6) == round(0.13, 6)
    assert round(tracker.avg_per_turn(), 6) == round(0.13, 6)

    summary = tracker.write_summary()
    assert summary["turns"] == 1
    assert summary["mode"] == "optimized"

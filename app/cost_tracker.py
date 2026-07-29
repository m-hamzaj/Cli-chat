"""Token/cost accounting for the chat session. Appends one JSON line per
API call to logs/costs.jsonl and writes a session summary on exit."""

import json
import os
import time

# USD per 1M tokens. Source: Groq pricing page, checked 2026-07-29.
PRICES = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "costs.jsonl")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "session_summary.json")


def price_call(model, input_tokens, output_tokens):
    rates = PRICES.get(model)
    if rates is None:
        # Unknown model: log but don't crash the session over a pricing gap.
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


class CostTracker:
    def __init__(self, session_id, mode):
        self.session_id = session_id
        self.mode = mode
        self.calls = []          # every API call, for the raw log
        self.turn_costs = []     # one entry per user turn (sum of its calls)
        self._current_turn_cost = 0.0
        self._current_turn_calls = 0
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    def record_call(self, model, input_tokens, output_tokens, purpose, turn_index):
        cost = price_call(model, input_tokens, output_tokens)
        entry = {
            "ts": time.time(),
            "session_id": self.session_id,
            "mode": self.mode,
            "turn": turn_index,
            "purpose": purpose,  # "tool_decision" | "response" | "final_after_tool"
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 8),
        }
        self.calls.append(entry)
        self._current_turn_cost += cost
        self._current_turn_calls += 1
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return cost

    def close_turn(self, turn_index):
        """Call once a user turn (all its API calls) is fully answered."""
        self.turn_costs.append({
            "turn": turn_index,
            "cost_usd": round(self._current_turn_cost, 8),
            "calls": self._current_turn_calls,
        })
        cost = self._current_turn_cost
        self._current_turn_cost = 0.0
        self._current_turn_calls = 0
        return cost

    def session_total(self):
        return sum(c["cost_usd"] for c in self.turn_costs)

    def avg_per_turn(self):
        if not self.turn_costs:
            return 0.0
        return self.session_total() / len(self.turn_costs)

    def write_summary(self):
        summary = {
            "session_id": self.session_id,
            "mode": self.mode,
            "turns": len(self.turn_costs),
            "total_cost_usd": round(self.session_total(), 8),
            "avg_cost_per_turn_usd": round(self.avg_per_turn(), 8),
            "per_turn": self.turn_costs,
        }
        # Keep a running list across sessions instead of clobbering the file.
        existing = []
        if os.path.exists(SUMMARY_PATH):
            try:
                with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(summary)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        return summary

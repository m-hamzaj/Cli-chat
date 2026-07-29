"""Repeat-question response cache. Keyed on (model, normalized question
text) -- NOT the full conversation transcript, because within a session the
transcript is never byte-identical between two askings of "the same"
question (each repeat has more prior history baked into it). Persisted to
disk so a repeat also short-circuits across separate sessions.

Tradeoff, worth being explicit about: this cache does not know whether the
conversation's context shifted between the two askings. "What's my name?"
asked twice will replay the first answer even if you told it a different
name in between. It trades a small chance of a stale answer for guaranteed
savings on literal repeats -- which is what "asked again" means in practice
for a CLI chat tool. A context-aware version would key on the full sent
transcript instead, but then it only ever fires on an exact restart with
identical history, which is rarely what "the same question again" means.
"""

import hashlib
import json
import os
import re

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "response_cache.json")


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def make_key(model, user_input):
    payload = f"{model}::{normalize(user_input)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self):
        self._data = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key):
        return self._data.get(key)

    def put(self, key, content):
        self._data[key] = content
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

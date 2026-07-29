"""Picks which model handles a turn in optimized mode.

Heuristic, applied BEFORE any API call (so a bad guess never costs an extra
round trip): short/plain messages and obvious tool requests go to the cheap
8B model; anything that looks like it needs real reasoning goes to 70B.
"""

import re

SMALL_MODEL = "llama-3.1-8b-instant"
BIG_MODEL = "llama-3.3-70b-versatile"

_COMPLEX_MARKERS = re.compile(
    r"\b(explain|analyze|analyse|compare|design|architecture|why does|why is|"
    r"pros and cons|write (a|an|some)|summarize|summarise|debug|refactor|"
    r"trade-?off|step by step|in detail|essay|plan)\b",
    re.IGNORECASE,
)


def choose_model(user_input, mode):
    if mode == "simple":
        return BIG_MODEL
    if len(user_input) > 200 or _COMPLEX_MARKERS.search(user_input):
        return BIG_MODEL
    return SMALL_MODEL

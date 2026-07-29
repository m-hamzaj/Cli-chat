# Groq CLI Chat

A command-line chat client for the Groq API. Streaming output, multi-turn
memory, two tools (calculator, fetch_url), and a per-call cost log — no
LangChain, no agent framework, just a hand-rolled request loop.

## Run it

```bash
cp .env.example .env        # put your real GROQ_API_KEY in .env
docker compose build
docker compose run --rm chat                # optimized mode (default)
docker compose run --rm chat --mode simple  # baseline mode, for comparison
```

Type `/cost` any time to see the running session total, `/exit` to quit.
Every session also writes `logs/costs.jsonl` (one line per API call) and
`logs/session_summary.json` (one entry per session, appended).

## Architecture

- `app/chat.py` — the REPL and the streaming request/response loop.
- `app/tools.py` — `calculator` (safe AST-based arithmetic eval, no `eval()`)
  and `fetch_url` (stdlib `urllib`, tag-stripped, truncated to 2000 chars).
  Both catch their own exceptions and return an error *string* — a bad
  expression or an unreachable URL becomes a tool-result message the model
  sees and responds to, not a crash.
- `app/router.py` — the model-choice heuristic used in optimized mode.
- `app/cost_tracker.py` — Groq's per-model pricing table and the JSONL logger.

Tool calls stream like everything else: one `stream=True` call is made per
step; if it finishes with `finish_reason == "tool_calls"`, the buffered
tool-call deltas are executed and a second streamed call produces the real
answer. Token usage comes from Groq's `stream_options={"include_usage": true}`
final chunk (passed via `extra_body`, since the installed SDK version doesn't
expose it as a named kwarg) — falls back to a `len(text)//4` estimate if a
response ever omits it.

## The two optimizations

**1. Model routing.** Every turn is priced up front — before the API call —
using a cheap heuristic in `router.py`: short, ordinary messages and obvious
tool requests go to `llama-3.1-8b-instant`; anything matching a "this needs
real reasoning" pattern (`explain`, `compare`, `in detail`, `summarize`,
`design`, ..., or just a long message) goes to `llama-3.3-70b-versatile`.
Groq's 8B model is ~11x cheaper per token than the 70B model, so getting most
turns onto the small model is the single biggest lever available.

**2. History trimming.** In optimized mode, the full conversation is still
kept in memory (so `/cost` and logs reflect everything), but only the last 10
messages are actually sent to the API each turn. `simple` mode sends the
entire, ever-growing history every turn.

## Results

Both numbers below come from the *same* 10-turn scripted conversation
(`demo_script.txt` — a mix of plain chat, calculator calls, `fetch_url`
calls, and one "explain in detail" turn), run for real against the Groq API,
logged to `logs/costs.jsonl`, and re-derived with `analyze.py`.

| | avg cost / turn | vs. baseline |
|---|---|---|
| **Before** — `--mode simple` (always 70B, full history resent) | **$0.001198** | — |
| **After** — `--mode optimized` (routed model + 10-message window) | **$0.000269** | **−77.5%** |

That's better than the "cut in half" bar. Which change did the work? I reran
the cost formula against the *actual logged token counts* with only one
change active at a time:

| Change in isolation | avg cost / turn | vs. baseline |
|---|---|---|
| History trimming only (still always 70B) | $0.000977 | −18.5% |
| Model routing only (still full history) | $0.000322 | −73.1% |
| Both together (what `optimized` mode does) | $0.000269 | −77.5% |

**Model routing is doing almost all of the work.** Groq charges ~11x more
per token for 70B than 8B, so simply keeping ordinary turns off the big model
dominates everything else — 8 of the 10 demo turns landed on 8B and cost
fractions of a cent each; the 2 that got routed to 70B (the "explain in
detail" turn and a later "summarize" turn) cost as much *by themselves* as
the other 8 combined.

**History trimming matters, but its dollar impact shrinks once you're mostly
on the cheap model.** Against an all-70B baseline it's worth 18.5% on its
own, because every token it removes was priced at 70B rates. Layered on top
of routing, the marginal savings are smaller in absolute cents (most
remaining traffic is already priced at $0.05/M input) — its real value there
is capping how large and slow prompts get as a conversation grows long, not
raw dollars.

**Tradeoff, measured, not hidden:** the 10-message window has a real cost.
In the demo run, turn 4 ("what's my name and what am I building?") and turn
10 ("what was the very first thing I told you?") both got wrong answers in
optimized mode — the name-introducing message had already scrolled out of
the window — while `simple` mode (full history) answered both correctly.
That's the accuracy/cost tradeoff this project asked for: a fixed window is
the cheapest form of memory management, not the smartest one. A production
version would summarize dropped turns into the system prompt instead of just
dropping them.

## What "no mistakes" required care around

- **Streaming + tool calls together**: tool-call argument deltas arrive
  chunked, same as content deltas — they're buffered by index and only
  parsed as JSON once the stream reports `finish_reason == "tool_calls"`.
- **Usage accounting under streaming**: Groq only emits token counts on a
  final, choice-less chunk when you opt in; missed that would silently zero
  out the cost log.
- **Tool errors are data, not exceptions**: `run_tool()` never raises: a
  division-by-zero, an unparseable expression (the model initially tried
  `"15% * 240"`, which isn't valid Python arithmetic — the tool description
  was tightened to tell it to convert percentages to decimals first), or an
  unreachable URL all become an error string fed back to the model, which
  then explains the problem to the user instead of the process dying.
- **Docker on Windows / Git Bash**: `docker run -v host:container` silently
  mistranslates container-side paths that look POSIX (`/app/logs`) unless
  `MSYS_NO_PATHCONV=1` is set — without it, the mount target gets rewritten
  and the log volume silently detaches (the app still prints correct
  in-memory numbers, so this fails quietly). Worth knowing if you run this
  from Git Bash on Windows instead of `docker compose`, which isn't affected.

## Files

```
app/chat.py           REPL, streaming loop, tool-call orchestration
app/tools.py           calculator, fetch_url, tool schemas
app/router.py          model-choice heuristic
app/cost_tracker.py    pricing table + JSONL/summary logging
demo_script.txt         the 10-turn conversation used for the before/after
analyze.py              recomputes the ablation table above from logs/costs.jsonl
Dockerfile, docker-compose.yml
logs/costs.jsonl        one line per API call (git-ignored in image, not in repo)
logs/session_summary.json
```

#!/usr/bin/env python3
"""CLI chat over the Groq API: streaming output, multi-turn memory, two
tools (calculator, fetch_url), and a per-call cost log.

Modes:
  --mode simple     always uses the 70B model, sends full history every turn.
  --mode optimized  (default) routes easy turns to the 8B model and caps how
                     much history is resent -- see router.py / MAX_HISTORY_MESSAGES.

An exact-prompt response cache (cache.py) also applies in both modes: if the
full sent context is byte-identical to something already answered, the
answer is replayed for free instead of calling the API again.
"""

import argparse
import json
import os
import sys
import uuid

from groq import Groq

from tools import TOOL_SCHEMAS, run_tool
from router import choose_model
from cost_tracker import CostTracker
from cache import ResponseCache, make_key

SYSTEM_PROMPT = (
    "You are a helpful, concise command-line assistant. You have exactly two "
    "tools: `calculator` for arithmetic, and `fetch_url` for reading a specific "
    "web page the user gives you a URL for. You do NOT have a search engine or "
    "any web-search tool -- never attempt to call one. For general-knowledge "
    "questions (facts, geography, history, definitions, etc.) answer directly "
    "from what you already know instead of trying to look anything up. "
    "If a tool returns an error, explain the problem briefly and keep going."
)

# Optimized mode only: cap how many prior messages get resent each turn.
# (System prompt is always sent in full and doesn't count against this.)
MAX_HISTORY_MESSAGES = 10


def estimate_tokens_from_text(text):
    return max(1, len(text or "") // 4)


def estimate_tokens_from_messages(messages):
    return sum(estimate_tokens_from_text(m.get("content")) for m in messages)


def windowed_history(history, mode):
    if mode == "simple":
        return history
    return history[-MAX_HISTORY_MESSAGES:]


MAX_TOOL_ROUNDS = 3  # hard cap on tool-call round-trips per turn, see handle_turn()


def run_completion(client, model, messages, cost_tracker, turn_index, purpose, allow_tools=True):
    """One streaming call. Prints content as it arrives, accumulates any
    tool-call deltas, and logs cost from real usage when the API returns it
    (falls back to a rough char/4 estimate if it doesn't). With
    allow_tools=False, no tools are offered at all, so the model has no
    option but to answer in plain text -- used to force a real final answer
    once MAX_TOOL_ROUNDS is hit."""
    tool_call_acc = {}
    content = ""
    finish_reason = None
    usage = None

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0.3,
        stream=True,
        extra_body={"stream_options": {"include_usage": True}},
    )
    if allow_tools:
        kwargs["tools"] = TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"
        # llama-3.1-8b-instant occasionally hallucinates a tool name (seen:
        # 'brave_search') that was never in TOOL_SCHEMAS. Without this, the
        # SDK raises on that instead of surfacing it as a normal tool call --
        # disabling lets it flow into run_tool()'s existing "unknown tool"
        # error path, which the model can see and recover from gracefully.
        kwargs["disable_tool_validation"] = True

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if delta and delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                slot = tool_call_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

    if usage:
        input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
    else:
        input_tokens = estimate_tokens_from_messages(messages)
        output_tokens = estimate_tokens_from_text(content)

    cost_tracker.record_call(model, input_tokens, output_tokens, purpose, turn_index)

    tool_calls = [v for _, v in sorted(tool_call_acc.items())]
    return content, tool_calls, finish_reason


def handle_turn(client, full_history, mode, cost_tracker, cache, turn_index, user_input):
    full_history.append({"role": "user", "content": user_input})
    model = choose_model(user_input, mode)
    sent = [{"role": "system", "content": SYSTEM_PROMPT}] + windowed_history(full_history, mode)

    cache_key = make_key(model, user_input)
    cached_answer = cache.get(cache_key)
    if cached_answer is not None:
        print(f"Assistant ({model}) [cache hit -- exact repeat of an earlier prompt]: {cached_answer}")
        full_history.append({"role": "assistant", "content": cached_answer})
        cost_tracker.record_call(model, 0, 0, "cache_hit", turn_index)
        turn_cost = cost_tracker.close_turn(turn_index)
        print(f"  (turn cost: ${turn_cost:.6f})")
        return

    final_answer = None
    for round_num in range(MAX_TOOL_ROUNDS + 1):
        force_final = round_num == MAX_TOOL_ROUNDS  # last chance: no tools offered, must answer in text
        purpose = "response" if round_num == 0 else ("forced_final" if force_final else "tool_round")

        print(f"Assistant ({model}): ", end="", flush=True)
        content, tool_calls, finish_reason = run_completion(
            client, model, sent, cost_tracker, turn_index, purpose, allow_tools=not force_final
        )

        if finish_reason != "tool_calls" or not tool_calls or force_final:
            final_answer = content
            full_history.append({"role": "assistant", "content": content})
            break

        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls if tc["id"]
            ],
        }
        full_history.append(assistant_msg)
        sent.append(assistant_msg)

        for tc in tool_calls:
            if not tc["id"]:
                continue
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            print(f"\n  [tool call: {tc['name']}({args})]", flush=True)
            result = run_tool(tc["name"], args)
            print(f"  [tool result: {result[:200]}]", flush=True)
            tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result}
            full_history.append(tool_msg)
            sent.append(tool_msg)

    if not final_answer:
        # Every round wanted a tool and even the forced no-tools round came
        # back empty -- surface something rather than an empty response.
        final_answer = "(no answer produced -- try rephrasing the question)"
        full_history[-1]["content"] = final_answer

    cache.put(cache_key, final_answer)

    print()
    turn_cost = cost_tracker.close_turn(turn_index)
    print(f"  (turn cost: ${turn_cost:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Groq-backed CLI chat tool")
    parser.add_argument("--mode", choices=["simple", "optimized"], default="optimized")
    parser.add_argument("--session-id", default=None, help="fixed session id, for reproducible demo runs")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set. Export it or put it in a .env file.", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)
    session_id = args.session_id or uuid.uuid4().hex[:8]
    cost_tracker = CostTracker(session_id, args.mode)
    cache = ResponseCache()
    full_history = []
    turn_index = 0
    interactive = sys.stdin.isatty()

    print(f"Groq CLI chat -- mode={args.mode}, session={session_id}")
    print("Commands: /cost to show running totals, /exit to quit.\n")

    try:
        while True:
            if interactive:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
            else:
                line = sys.stdin.readline()
                if not line:
                    break
                user_input = line.strip()
                if not user_input:
                    continue
                print(f"You: {user_input}")

            if not user_input:
                continue
            if user_input in ("/exit", "exit", "quit"):
                break
            if user_input == "/cost":
                print(f"  session total: ${cost_tracker.session_total():.6f} "
                      f"over {len(cost_tracker.turn_costs)} turn(s), "
                      f"avg/turn: ${cost_tracker.avg_per_turn():.6f}")
                continue

            turn_index += 1
            try:
                handle_turn(client, full_history, args.mode, cost_tracker, cache, turn_index, user_input)
            except Exception as e:
                # A bad API/tool interaction should not kill the session.
                print(f"\n  [error handling turn: {e}]")
    except KeyboardInterrupt:
        print("\n(interrupted)")

    summary = cost_tracker.write_summary()
    print("\n--- session summary ---")
    print(f"mode: {summary['mode']}  turns: {summary['turns']}  "
          f"total: ${summary['total_cost_usd']:.6f}  "
          f"avg/turn: ${summary['avg_cost_per_turn_usd']:.6f}")
    print(f"Full log: logs/costs.jsonl | Summary: logs/session_summary.json")


if __name__ == "__main__":
    main()

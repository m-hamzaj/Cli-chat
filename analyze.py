import json, collections

def load(session_id):
    rows = []
    with open("logs/costs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["session_id"] == session_id:
                rows.append(r)
    return rows

simple = load("demo-simple")
opt = load("demo-optimized")

PRICES = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}
def price(model, i, o):
    r = PRICES[model]
    return (i/1_000_000)*r["input"] + (o/1_000_000)*r["output"]

def turn_totals(rows):
    t = collections.defaultdict(lambda: {"in":0,"out":0,"cost":0.0,"models":set()})
    for r in rows:
        t[r["turn"]]["in"] += r["input_tokens"]
        t[r["turn"]]["out"] += r["output_tokens"]
        t[r["turn"]]["cost"] += r["cost_usd"]
        t[r["turn"]]["models"].add(r["model"])
    return t

st = turn_totals(simple)
ot = turn_totals(opt)

print("=== BASELINE (simple mode: always 70B, full history) ===")
tot=0
for n in sorted(st):
    tot+=st[n]["cost"]
    print(f"turn {n}: in={st[n]['in']} out={st[n]['out']} cost=${st[n]['cost']:.6f} model={st[n]['models']}")
print(f"TOTAL=${tot:.6f} AVG/TURN=${tot/len(st):.6f}\n")

print("=== OPTIMIZED (routed model + 10-msg history window) ===")
tot2=0
for n in sorted(ot):
    tot2+=ot[n]["cost"]
    print(f"turn {n}: in={ot[n]['in']} out={ot[n]['out']} cost=${ot[n]['cost']:.6f} model={ot[n]['models']}")
print(f"TOTAL=${tot2:.6f} AVG/TURN=${tot2/len(ot):.6f}\n")

trim_only = 0.0
for r in opt:
    trim_only += price("llama-3.3-70b-versatile", r["input_tokens"], r["output_tokens"])
print(f"Ablation A - trimming only (trimmed tokens, forced 70B): total=${trim_only:.6f} avg/turn=${trim_only/len(ot):.6f}")

route_only = 0.0
opt_model_by_turn_call = {}
for r in opt:
    opt_model_by_turn_call.setdefault(r["turn"], []).append(r["model"])
for r in simple:
    calls_for_turn = opt_model_by_turn_call.get(r["turn"], ["llama-3.3-70b-versatile"])
    model = calls_for_turn[0]
    route_only += price(model, r["input_tokens"], r["output_tokens"])
n_turns = len(st)
print(f"Ablation B - routing only (full-history tokens, routed model): total=${route_only:.6f} avg/turn=${route_only/n_turns:.6f}")

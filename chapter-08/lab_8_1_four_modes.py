"""Lab 8.1 (Section 8.6): run the Chapter 7 service assistant in four deployment modes
and fill in Table 8.6.

    python lab_8_1_four_modes.py stub                  # offline check of the harness
    python lab_8_1_four_modes.py api                   # Mode 1, commercial API
    python lab_8_1_four_modes.py inprocess             # Mode 2, Transformers in this process
    python lab_8_1_four_modes.py ollama                # Mode 3, Ollama on your laptop
    python lab_8_1_four_modes.py vllm --gpu-hourly 2.0 # Mode 4, vLLM on a rented GPU
    python lab_8_1_four_modes.py api ollama vllm       # several modes in one go

Set model names in backends.py first. For Mode 1 cost, pass your provider's current
prices with --price-in and --price-out (US dollars per million tokens).

Each mode writes results_<mode>.csv (one row per question); the summary is appended to
summary.csv so that runs on different machines can be combined.
"""
import argparse, csv, os, re, statistics, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "chapter-07"))
from rag_pipeline import kb, hybrid_search, build_messages     # the Chapter 7 pipeline, unchanged
from backends import get_backend, call_with_retries

NOT_FOUND = "I could not find this in the service documents."
WHERE = {"api": "Provider", "inprocess": "Your machine", "ollama": "Your machine",
         "vllm": "Your cloud account", "stub": "Your machine"}

p = argparse.ArgumentParser()
p.add_argument("modes", nargs="+", choices=["stub", "api", "inprocess", "ollama", "vllm"])
p.add_argument("--limit", type=int, default=None, help="use only the first N questions")
p.add_argument("--price-in", type=float, default=None, help="USD per million input tokens (Mode 1)")
p.add_argument("--price-out", type=float, default=None, help="USD per million output tokens (Mode 1)")
p.add_argument("--gpu-hourly", type=float, default=None, help="USD per GPU hour (Mode 4)")
args = p.parse_args()

questions = [(q["q"], False) for q in kb["questions"]]
questions += [(line.strip(), True) for line in open("not_in_kb.txt", encoding="utf-8") if line.strip()]
if args.limit:
    questions = questions[:args.limit]

def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None

summary = []
for mode in args.modes:
    call = get_backend(mode)
    rows = []
    for i, (q, absent) in enumerate(questions, 1):
        hits = hybrid_search(q)[:4]
        r = call_with_retries(call, build_messages(q, hits), max_tokens=300, temperature=0)
        rows.append({"question": q, "not_in_kb": absent, "answer": r["text"],
                     "cites_source": bool(re.search(r"\[\d+\]", r["text"])),
                     "said_not_found": NOT_FOUND.lower()[:30] in r["text"].lower(),
                     "ttft_s": r["ttft_s"], "total_s": r["total_s"],
                     "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
        print(f"\r{mode}: {i}/{len(questions)}", end="", flush=True)
    print()
    with open(f"results_{mode}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    present = [r for r in rows if not r["not_in_kb"]]
    absent = [r for r in rows if r["not_in_kb"]]
    tin = sum(r["input_tokens"] or 0 for r in rows); tout = sum(r["output_tokens"] or 0 for r in rows)
    cost = None
    if mode == "api" and args.price_in is not None and args.price_out is not None:
        cost = (tin * args.price_in + tout * args.price_out) / 1e6 / len(rows) * 1000
    if mode == "vllm" and args.gpu_hourly:
        per_hour = 3600 / statistics.mean(r["total_s"] for r in rows)   # one user, back to back
        cost = args.gpu_hourly / per_hour * 1000
    summary.append({"mode": mode,
        "cites_source_pct": round(100 * sum(r["cites_source"] for r in present) / max(1, len(present))),
        "correct_not_found_pct": round(100 * sum(r["said_not_found"] for r in absent) / len(absent)) if absent else None,
        "median_ttft_s": median([r["ttft_s"] for r in rows]),
        "median_total_s": median([r["total_s"] for r in rows]),
        "input_tokens": tin, "output_tokens": tout,
        "cost_per_1000_questions_usd": round(cost, 2) if cost is not None else None,
        "documents_went_to": WHERE[mode]})

new = not os.path.exists("summary.csv")
with open("summary.csv", "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    if new: w.writeheader()
    w.writerows(summary)

labels = [("cites_source_pct", "Answers citing a source (%)"),
          ("correct_not_found_pct", 'Correct "not found" replies (%)'),
          ("median_ttft_s", "Median time to first token (s)"),
          ("median_total_s", "Median total time (s)"),
          ("cost_per_1000_questions_usd", "Estimated cost per 1,000 questions"),
          ("documents_went_to", "Where the documents went")]
print(f"\n{'Measure':36s}" + "".join(f"{s['mode']:>16s}" for s in summary))
for key, label in labels:
    print(f"{label:36s}" + "".join(f"{str(s[key] if s[key] is not None else '-'):>16s}" for s in summary))
print("\nFaithfulness: score results_<mode>.csv with the RAGAS script from chapter-07, Part C.")

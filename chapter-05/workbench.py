"""Prompt engineering workbench: run prompt versions on an evaluation set and compare them."""
import json, time, datetime, pathlib
from load_prompt import load_prompt, render
from structured import parse_claim

FIELDS = ["policy_number", "claim_type", "incident_date"]

def load_eval(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

def run_version(prompt_path, dataset, call_fn, log_dir="runs"):
    spec = load_prompt(prompt_path)
    results = []
    for ex in dataset:
        messages = render(spec, email=ex["email"],
                          claim_types=["MOTOR", "HEALTH", "PROPERTY", "TRAVEL"])
        t0 = time.perf_counter()
        reply = call_fn(messages, model=spec["model"], temperature=spec["temperature"],
                        max_tokens=spec["max_tokens"])
        latency = time.perf_counter() - t0
        claim, error = parse_claim(reply["text"])
        got = claim.model_dump(mode="json") if claim else {}
        correct = {f: got.get(f) == ex["expected"][f] for f in FIELDS} if claim else {f: False for f in FIELDS}
        results.append({"id": ex["id"], "valid": claim is not None, "error": error,
                        "correct": correct, "latency_s": round(latency, 3),
                        "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"]})
    # keep a permanent, traceable record of every run
    pathlib.Path(log_dir).mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(log_dir) / f"{spec['name']}-{spec['version']}-{spec['fingerprint']}-{stamp}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
    return spec, results

def summarise(spec, results):
    n = len(results)
    row = {"version": spec["version"], "valid_json": sum(r["valid"] for r in results) / n}
    for f in FIELDS:
        row[f] = sum(r["correct"][f] for r in results) / n
    row["all_fields"] = sum(all(r["correct"].values()) for r in results) / n
    row["avg_in_tokens"] = sum(r["input_tokens"] for r in results) / n
    row["avg_out_tokens"] = sum(r["output_tokens"] for r in results) / n
    return row

def compare(results_a, results_b):
    """Paired comparison: which examples did B fix, and which did it break?"""
    a = {r["id"]: all(r["correct"].values()) for r in results_a}
    b = {r["id"]: all(r["correct"].values()) for r in results_b}
    fixed = [i for i in a if not a[i] and b[i]]
    broken = [i for i in a if a[i] and not b[i]]
    return fixed, broken

if __name__ == "__main__":
    import sys
    from providers import offline_stub, call_anthropic, call_openai_compatible
    backend = {"stub": offline_stub, "anthropic": call_anthropic,
               "openai": call_openai_compatible}[sys.argv[1] if len(sys.argv) > 1 else "stub"]
    data = load_eval("claims_eval.jsonl")
    spec1, r1 = run_version("prompts/claim_extractor_v1.yaml", data, backend)
    spec2, r2 = run_version("prompts/claim_extractor_v2.yaml", data, backend)
    for row in (summarise(spec1, r1), summarise(spec2, r2)):
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    fixed, broken = compare(r1, r2)
    print("v2 fixed:", fixed, " v2 broke:", broken)

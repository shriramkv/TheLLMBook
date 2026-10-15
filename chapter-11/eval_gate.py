"""
eval_gate.py  -  A release gate for an LLM service.

Chapter 11, LLMOps. A deployment should not ship because it "looks fine"; it
ships because it passed a gate: a fixed evaluation set with thresholds that
must be met. This script is that gate, made self-contained and reproducible.

It runs three checks against a set of recorded answers:
  - task quality:   fraction of answers that contain the expected content
  - regression:     no previously-passing case may now fail
  - honest refusal: unanswerable questions must be refused, not guessed

It exits 0 (ship) only if every threshold is met, and non-zero (block) otherwise,
so it can be wired straight into CI. The example set runs offline; point it at
your own service's recorded answers to use it for real.
"""
import json
import sys

# --- Thresholds that define "good enough to ship". Tune for your system. ---
MIN_TASK_QUALITY = 0.80      # at least 80% of answerable cases correct
MIN_REFUSAL_RATE = 0.90      # at least 90% of unanswerable cases refused
ALLOW_REGRESSIONS = 0        # zero previously-passing cases may regress

# --- An illustrative eval set. Replace `answer` with your service's output. ---
EVAL_SET = [
    {"id": "q1", "question": "What is the refund window?",
     "expect": ["30 days"], "answer": "Refund requests are accepted within 30 days.", "answerable": True},
    {"id": "q2", "question": "How do I reset my password?",
     "expect": ["reset", "email"], "answer": "Use the reset link sent to your email.", "answerable": True},
    {"id": "q3", "question": "What is the warranty period?",
     "expect": ["12 months"], "answer": "The warranty covers 12 months from purchase.", "answerable": True},
    {"id": "q4", "question": "Do you ship to Antarctica?",
     "expect": [], "answer": "I could not find that information in the provided documents.",
     "answerable": False},
    {"id": "q5", "question": "What is the CEO's home address?",
     "expect": [], "answer": "I can't answer that from the knowledge base.", "answerable": False},
]

# A record of which cases passed on the last shipped release (for regression).
LAST_RELEASE_PASSED = {"q1", "q2", "q3", "q4", "q5"}

REFUSAL_MARKERS = ["could not find", "can't answer", "cannot answer", "not in the", "don't know"]


def is_correct(case):
    text = case["answer"].lower()
    if case["answerable"]:
        return all(kw.lower() in text for kw in case["expect"])
    return any(m in text for m in REFUSAL_MARKERS)


def main():
    answerable = [c for c in EVAL_SET if c["answerable"]]
    unanswerable = [c for c in EVAL_SET if not c["answerable"]]

    passed = {c["id"] for c in EVAL_SET if is_correct(c)}

    task_quality = sum(is_correct(c) for c in answerable) / max(len(answerable), 1)
    refusal_rate = sum(is_correct(c) for c in unanswerable) / max(len(unanswerable), 1)
    regressions = sorted(LAST_RELEASE_PASSED - passed)

    print("=== Release gate ===")
    print(f"Task quality (answerable):  {task_quality:.2f}  (threshold >= {MIN_TASK_QUALITY})")
    print(f"Honest refusal (unanswerable): {refusal_rate:.2f}  (threshold >= {MIN_REFUSAL_RATE})")
    print(f"Regressions vs last release: {len(regressions)}  (allowed {ALLOW_REGRESSIONS})")
    if regressions:
        print("  regressed cases:", ", ".join(regressions))

    ok = (task_quality >= MIN_TASK_QUALITY
          and refusal_rate >= MIN_REFUSAL_RATE
          and len(regressions) <= ALLOW_REGRESSIONS)

    result = {"task_quality": task_quality, "refusal_rate": refusal_rate,
              "regressions": regressions, "ship": ok}
    with open("eval_gate_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nVERDICT:", "SHIP" if ok else "BLOCK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

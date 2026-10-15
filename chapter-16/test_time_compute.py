"""
test_time_compute.py  -  Why does spending more compute at inference help,
and when does it stop? A reproducible, exact demonstration.

Chapter 16, Section 16.2 (Reasoning Models and Test-Time Compute).

"Self-consistency" samples a model's answer N times and takes the majority
vote. Its behaviour is governed by the Condorcet jury theorem: if each
independent attempt is correct with probability p, the majority of N attempts
is correct with a probability we can compute exactly from the binomial
distribution. This script computes that probability, with no randomness, so it
reproduces identically on any machine, and it reveals the promise AND the hard
limit of test-time compute.

Key result (see output):
  - If p > 0.5, majority accuracy rises towards 1 as N grows, with diminishing returns.
  - If p < 0.5, majority accuracy FALLS towards 0: more compute makes it worse.
  - At p = 0.5, it stays at 0.5 forever.
So test-time compute amplifies a model that is better than chance and amplifies
the error of one that is not. It buys accuracy with compute only past a threshold.
"""
import json
from math import comb


def majority_correct(p, n):
    """Exact probability that the majority of n independent attempts is correct,
    each correct with probability p. Ties (even n, equal split) broken 50/50."""
    total = 0.0
    for k in range(n + 1):
        prob_k = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        if k > n / 2:
            total += prob_k
        elif k == n / 2:  # a tie, only for even n
            total += 0.5 * prob_k
    return total


def main():
    ns = [1, 3, 5, 7, 11, 15, 21, 31]
    ps = [0.40, 0.50, 0.55, 0.65, 0.80]

    print("Self-consistency: majority-vote accuracy by per-sample accuracy p and samples N")
    print("(exact, from the binomial distribution; the Condorcet jury theorem)\n")
    header = "p \\ N  " + "".join(f"{n:>8d}" for n in ns)
    print(header)
    curves = {}
    for p in ps:
        row = [majority_correct(p, n) for n in ns]
        curves[p] = row
        print(f"{p:<6.2f} " + "".join(f"{v:>8.3f}" for v in row))

    # The cost view: accuracy gained per doubling of compute, for a good model.
    print("\nDiminishing returns for p = 0.65 (accuracy gained per extra samples):")
    good = curves[0.65]
    for i in range(1, len(ns)):
        dn = ns[i] - ns[i - 1]
        dacc = good[i] - good[i - 1]
        print(f"  N {ns[i-1]:>2d} -> {ns[i]:>2d}  (+{dn} samples): accuracy +{dacc:.4f}")

    # The threshold, shown plainly.
    print("\nAt N = 31:")
    for p in ps:
        verdict = "amplified upward" if p > 0.5 else ("amplified downward" if p < 0.5 else "unchanged")
        print(f"  p = {p:.2f}  ->  majority accuracy {curves[p][-1]:.3f}  ({verdict})")

    with open("ttc_results.json", "w") as f:
        json.dump({"ns": ns, "ps": ps, "curves": {str(p): curves[p] for p in ps}}, f, indent=2)
    print("\nWrote ttc_results.json")


if __name__ == "__main__":
    main()

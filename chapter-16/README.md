# Chapter 16: The Future of Generative AI

One experiment: the exact mathematics of test-time compute.

## Files

- `test_time_compute.py` — computes, exactly from the binomial distribution, the accuracy of self-consistency (majority vote over N samples) as a function of the per-sample accuracy p. No randomness, so it reproduces identically. Standard library only (`math`).

## Run

```bash
python3 test_time_compute.py
```

`ttc_results.json` holds the book's run. The result is the Condorcet jury theorem applied to self-consistency: if each attempt is better than chance (p above 0.5), more samples drive accuracy towards 1 with diminishing returns; if each attempt is worse than chance (p below 0.5), more samples drive it towards 0; at exactly chance it never moves.

The lesson for reasoning models: test-time compute is a **multiplier** on per-attempt accuracy, not a substitute for it. Measure the per-sample accuracy first; if it is below chance, no amount of extra compute will save you.

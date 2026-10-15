# Chapter 13: Ethics, Governance and Regulation

One experiment: measuring bias in embeddings with the Word Embedding Association Test (WEAT).

## Files

- `weat_experiment.py` — runs WEAT on the WordLlama embeddings used in Chapter 7, with a permutation test for significance. It reports an effect size and a p-value for each association. Needs `wordllama` and `numpy`.

## Run

```bash
python3 weat_experiment.py
```

`weat_results.json` holds the book's run. The flower-versus-insect sanity check comes out large and significant (d around 1.39), which shows the method works; the social-association tests are weaker on this small model. The point is the **method**: choose the groups and attributes that matter for your application, run it on whichever embedding model you deploy, and track it as you change models.

This experiment teaches a measurement technique. A single run on one small model is not a verdict about people; it is a tool for watching your own system.

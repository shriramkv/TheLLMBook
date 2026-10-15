# Chapter 15: Capstone Projects

The measured core of Capstone 1: does advanced retrieval actually beat a dense baseline?

## Files

- `hybrid_rag_experiment.py` — builds a 24-chunk enterprise-policy corpus, indexes it with WordLlama (dense) and BM25 (lexical), fuses them with Reciprocal Rank Fusion, and scores all three on 34 labelled questions split into paraphrase and exact-term types. Reports hit-rate by rank and by query type. Needs `wordllama`, `rank_bm25` and `numpy`.

## Run

```bash
python3 hybrid_rag_experiment.py
```

`hybrid_results.json` holds the book's run. The honest finding: **no retriever wins everywhere**. Dense leads on paraphrased questions (meaning matters more than words); lexical and hybrid lead on exact-term questions, including a form-code query where the four near-identical form descriptions are semantically almost the same and only the code separates them. Hybrid is never the worst of the three.

The deliverable of a real capstone is this harness, not a fixed choice of retriever: measure on your own query mix and choose from the numbers.

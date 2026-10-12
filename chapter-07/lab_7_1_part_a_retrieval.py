"""Lab 7.1, Part A (Section 7.7): measure keyword, dense and hybrid retrieval on the
Nilgiri Flow Systems knowledge base, and the effect of chunk size and title prefixes.

Reproduces Tables 7.5 and 7.6 and the data for Figure 7.2. Takes under a minute on a laptop.
    pip install rank_bm25 numpy wordllama
    python lab_7_1_part_a_retrieval.py
"""
import json
import numpy as np
from rank_bm25 import BM25Okapi
from rag_pipeline import kb, tokens, chunk, wl

def embed(texts):
    v = np.asarray(wl.embed(texts), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def run(size, overlap, title=True):
    """Index the knowledge base and record, for every question and retriever, the rank of
    the first chunk that comes from a correct document (None if it never appears)."""
    chunks = [c for d in kb["docs"] for c in chunk(d, size, overlap, title)]
    bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
    E = embed([c["text"] for c in chunks])
    res = {}
    for q in kb["questions"]:
        r_bm = np.argsort(-bm25.get_scores(tokens(q["q"])))
        r_de = np.argsort(-(E @ embed([q["q"]])[0]))
        rrf = np.zeros(len(chunks))
        for ranks in (r_bm, r_de):
            for pos, i in enumerate(ranks):
                rrf[i] += 1 / (60 + pos + 1)
        for name, order in (("BM25", r_bm), ("Dense", r_de), ("Hybrid (RRF)", np.argsort(-rrf))):
            docs_ranked = []                       # several chunks can share a document
            for i in order:
                if chunks[i]["doc"] not in docs_ranked:
                    docs_ranked.append(chunks[i]["doc"])
            first = next((p for p, d in enumerate(docs_ranked) if d in q["gold"]), None)
            res.setdefault(name, []).append((q["type"], first))
    return len(chunks), res

def hit(rows, k):
    return round(100 * np.mean([f is not None and f < k for _, f in rows]))

def summarise(res):
    out = {}
    for name, rows in res.items():
        mrr = round(float(np.mean([0 if f is None else 1 / (f + 1) for _, f in rows])), 2)
        out[name] = {"hit@1": hit(rows, 1), "hit@3": hit(rows, 3), "hit@10": hit(rows, 10), "mrr": mrr,
                     **{t: hit([r for r in rows if r[0] == t], 3) for t in ("exact", "paraphrase", "general")}}
    return out

n, main = run(50, 10, True)
s = summarise(main)
print(f"Table 7.5  ({len(kb['questions'])} questions, {n} chunks of 50 words with titles)")
print(f"{'Retriever':14s} {'Hit@1':>6s} {'Hit@3':>6s} {'Hit@10':>7s} {'MRR':>5s}")
for name, v in s.items():
    print(f"{name:14s} {v['hit@1']:5d}% {v['hit@3']:5d}% {v['hit@10']:6d}% {v['mrr']:5.2f}")

print("\nFigure 7.2  hit at 3 by question type")
for name, v in s.items():
    print(f"{name:14s} exact {v['exact']:3d}%   paraphrase {v['paraphrase']:3d}%   general {v['general']:3d}%")

print("\nTable 7.6  chunk size and title prefixes (hit at 3)")
sizes = {}
for label, size, ov, title in (("20-word chunks, with titles", 20, 5, True),
                               ("50-word chunks, with titles", 50, 10, True),
                               ("120-word chunks, with titles", 120, 20, True),
                               ("50-word chunks, without titles", 50, 10, False)):
    n, r = run(size, ov, title)
    t = summarise(r)
    sizes[label] = {"chunks": n, **{m: t[m]["hit@3"] for m in t}}
    print(f"{label:32s} {n:4d} chunks  " + "  ".join(f"{m} {t[m]['hit@3']}%" for m in t))

json.dump({"main": s, "chunk_sizes": sizes}, open("retrieval_results.json", "w"), indent=1)
print("\nSaved retrieval_results.json")

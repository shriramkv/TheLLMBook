"""Chapter 13, Section 13.1: measuring representational bias in embeddings (a WEAT-style test).

Replicates the method of Caliskan, Bryson and Narayanan (2017): measure whether two sets of TARGET
words (e.g. career vs family) are more associated with one or the other of two ATTRIBUTE sets
(e.g. traditionally male vs female given names) in an embedding space. The effect size d and a
permutation p-value quantify the association. We use the WordLlama embeddings from Chapter 7, so the
whole test runs offline. This measures the embedding model's associations, not any truth about people.
"""
import numpy as np, itertools, json
from wordllama import WordLlama
wl = WordLlama.load()
def vecs(words):
    v = np.asarray(wl.embed(words), dtype=np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True)
def cos(a, b): return a @ b.T

TESTS = {
 "career_vs_family": {
   "X": ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"],
   "Y": ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"],
   "A": ["john", "paul", "mike", "kevin", "steve", "greg", "jeff", "bill"],
   "B": ["amy", "joan", "lisa", "sarah", "diana", "kate", "ann", "donna"]},
 "science_vs_arts": {
   "X": ["science", "technology", "physics", "chemistry", "engineering", "mathematics", "experiment", "astronomy"],
   "Y": ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
   "A": ["brother", "father", "uncle", "grandfather", "son", "he", "his", "man"],
   "B": ["sister", "mother", "aunt", "grandmother", "daughter", "she", "her", "woman"]},
 "pleasant_insect_flower": {   # a sanity check: flowers pleasant, insects unpleasant (a very robust effect)
   "X": ["rose", "tulip", "lily", "daisy", "orchid", "jasmine", "marigold", "lotus"],
   "Y": ["ant", "cockroach", "mosquito", "spider", "locust", "termite", "wasp", "maggot"],
   "A": ["joy", "love", "peace", "wonderful", "pleasure", "friend", "happy", "gentle"],
   "B": ["agony", "terrible", "horrible", "nasty", "evil", "awful", "disaster", "pain"]},
}
def weat(t, n_perm=10000, seed=0):
    X, Y, A, B = (vecs(t[k]) for k in "XYAB")
    def s(w): return cos(w, A).mean() - cos(w, B).mean()
    sx, sy = np.array([s(w) for w in X]), np.array([s(w) for w in Y])
    d = (sx.mean() - sy.mean()) / np.concatenate([sx, sy]).std(ddof=1)
    allw = np.concatenate([sx, sy]); obs = sx.sum() - sy.sum(); rng = np.random.default_rng(seed)
    k = len(sx); cnt = sum((lambda p: p[:k].sum() - p[k:].sum() > obs)(rng.permutation(allw)) for _ in range(n_perm))
    return round(float(d), 2), round((cnt + 1) / (n_perm + 1), 4)
out = {}
for name, t in TESTS.items():
    d, p = weat(t); out[name] = {"effect_size_d": d, "p_value": p}
    print(f"{name:26s} d = {d:+.2f}   p = {p}")
json.dump(out, open("weat_results.json", "w"), indent=1)
print("\nd > 0 means the X targets are more associated with set A than the Y targets are.")

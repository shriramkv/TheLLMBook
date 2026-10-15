"""Chapter 11, Section 11.3: how often does a semantic cache answer correctly, and how often wrongly?
36 cached questions from the Chapter 7 knowledge base. For each: one paraphrase that should hit the
cache (same answer) and one near-miss that should NOT (a different part, code or quantity)."""
import json, re, numpy as np
from wordllama import WordLlama
pairs = json.load(open("cache_pairs.json"))
wl = WordLlama.load()
def emb(t):
    v = np.asarray(wl.embed(t), dtype=np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True)
cached, para, near = (emb([p[i] for p in pairs]) for i in range(3))
def norm(t): return re.sub(r"[^a-z0-9 -]", "", t.lower()).strip()
exact_para = np.mean([norm(p[1]) == norm(p[0]) for p in pairs]); exact_near = np.mean([norm(p[2]) == norm(p[0]) for p in pairs])
def best(qs):  # nearest cached question and its similarity
    s = qs @ cached.T; return s.argmax(1), s.max(1)
pi, ps = best(para); ni, ns = best(near)
rows = []
for th in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    ok = np.mean((ps >= th) & (pi == np.arange(len(pairs))))            # paraphrase served the right answer
    wrong_para = np.mean((ps >= th) & (pi != np.arange(len(pairs))))    # paraphrase served someone else's answer
    wrong_near = np.mean(ns >= th)                                        # near-miss served a cached (wrong) answer
    rows.append(dict(threshold=th, correct_hits=round(100*ok), wrong_hits_paraphrase=round(100*wrong_para), wrong_hits_near_miss=round(100*wrong_near)))
    print(rows[-1])
print("exact-match cache: paraphrase hits", exact_para, "near-miss hits", exact_near)
print("similarity: paraphrases mean %.3f median %.3f | near-misses mean %.3f median %.3f" % (ps.mean(), np.median(ps), ns.mean(), np.median(ns)))
print("near-misses more similar than their own paraphrase:", int(np.sum(ns > ps)), "of", len(pairs))
for k in np.argsort(-ns)[:5]: print("  %.3f  %s  ->  %s" % (ns[k], pairs[k][2], pairs[ni[k]][0]))
json.dump(dict(rows=rows, para_sim=ps.round(3).tolist(), near_sim=ns.round(3).tolist()), open("cache_results.json", "w"), indent=1)

# A guard: only serve a cached answer if the identifiers (codes, model numbers) match exactly.
codes = lambda t: frozenset(re.findall(r"\b[A-Z]{1,3}(?:-[A-Z]{1,3})*-\d+\b", t))
guard_rows = []
for th in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    okp = [(ps[i] >= th) and pi[i] == i and codes(pairs[i][1]) == codes(pairs[pi[i]][0]) for i in range(len(pairs))]
    badn = [(ns[i] >= th) and codes(pairs[i][2]) == codes(pairs[ni[i]][0]) for i in range(len(pairs))]
    guard_rows.append(dict(threshold=th, correct_hits=round(100*np.mean(okp)), wrong_hits_near_miss=round(100*np.mean(badn))))
    print("guarded", guard_rows[-1])
n_code = sum(1 for p in pairs if codes(p[0]) != codes(p[2]))
print("near-misses that differ by an identifier:", n_code, "of", len(pairs))
res = json.load(open("cache_results.json")); res["guarded"] = guard_rows; json.dump(res, open("cache_results.json", "w"), indent=1)

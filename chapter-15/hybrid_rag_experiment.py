"""
hybrid_rag_experiment.py  -  Does "advanced RAG" actually beat naive RAG?

Chapter 15, Capstone 1 (Enterprise Knowledge Assistant). We measure retrieval
quality for three strategies on the same corpus and the same questions:

  1. Dense only   - WordLlama embeddings, cosine similarity (the Chapter 7 baseline)
  2. Lexical only - BM25 (rank_bm25)
  3. Hybrid       - the two fused with Reciprocal Rank Fusion (RRF)

Offline and reproducible: WordLlama and rank_bm25 are the same components used
earlier in the book, no API keys, no network at run time. The corpus is a small
set of enterprise-policy snippets; each question has a known relevant chunk, so
we can score hit-rate@k and Mean Reciprocal Rank. The point is the METHOD and the
comparison, not the absolute numbers, which depend on the corpus size.
"""
import json
import numpy as np
from wordllama import WordLlama
from rank_bm25 import BM25Okapi

# --- A small enterprise-policy corpus. Each entry is one retrievable chunk. ---
CHUNKS = [
    ("leave-annual", "Employees accrue 1.5 days of paid annual leave for each completed month of service, up to a maximum of 18 days per calendar year."),
    ("leave-carry", "Unused annual leave of up to 6 days may be carried forward to the next calendar year; any excess lapses on 31 March."),
    ("leave-sick", "Sick leave is granted up to 12 days per year and requires a medical certificate for any absence longer than two consecutive days."),
    ("leave-maternity", "Maternity leave of 26 weeks is available to eligible employees, with the first 8 weeks taken before the expected date of delivery if requested."),
    ("expense-travel", "Domestic travel is reimbursed at actuals for economy airfare and up to Rs 4000 per night for hotel accommodation in metro cities."),
    ("expense-claim", "Expense claims must be submitted within 30 days of the expense with original receipts; claims submitted after 60 days will not be reimbursed."),
    ("expense-perdiem", "A per diem of Rs 1200 per day covers meals and incidentals during domestic travel and does not require individual receipts."),
    ("it-password", "Corporate passwords must be at least 12 characters, changed every 90 days, and must not be reused across the last 5 passwords."),
    ("it-vpn", "Remote access to internal systems requires the corporate VPN and multi-factor authentication; access from personal devices needs prior approval."),
    ("it-data", "Customer personal data must not be copied to personal cloud storage or pasted into third-party tools that are not on the approved list."),
    ("procure-po", "Any purchase above Rs 50000 requires a purchase order approved by the department head before the commitment is made."),
    ("procure-vendor", "New vendors must clear a due-diligence check and be added to the vendor master before any payment can be processed."),
    ("notice-period", "The notice period for voluntary resignation is 60 days for managers and 30 days for all other employees, served or bought out."),
    ("probation", "New employees serve a probation period of 6 months, during which either party may end employment with 15 days of notice."),
    ("wfh-policy", "Employees may work from home up to 2 days per week with manager approval; fully remote arrangements require a formal exception."),
    ("reimburse-internet", "A home internet allowance of Rs 1000 per month is available to employees on an approved work-from-home arrangement."),
    ("training-budget", "Each employee has an annual learning budget of Rs 25000 for approved courses, certifications and conferences."),
    ("grievance", "Workplace grievances may be raised with the employee's manager or, if unresolved, escalated to the HR grievance committee in writing."),
    ("asset-return", "All company assets, including laptops and access cards, must be returned on the last working day before the final settlement is released."),
    ("referral-bonus", "A referral bonus of Rs 30000 is paid if a referred candidate is hired and completes 6 months of continuous service."),
    # A cluster of near-identical forms, distinguished only by a code and one noun.
    # This is where dense retrieval struggles: the sentences are semantically almost
    # the same, so the embedding cannot separate them; the code is the disambiguator.
    ("form-leave", "Form HR-12 is the application for leave encashment and must be approved by the reporting manager."),
    ("form-travel", "Form HR-21 is the application for a travel advance and must be approved by the reporting manager."),
    ("form-vpn", "Form IT-07 is the request for VPN access and must be approved by the reporting manager."),
    ("form-asset", "Form IT-09 is the request for a replacement laptop and must be approved by the reporting manager."),
]

# --- Questions, each with the id of the one relevant chunk and a query TYPE.
# "paraphrase" rewords the need in everyday language (favours dense/semantic).
# "lexical" uses the exact jargon, acronyms or figures a real user would type
# when they half-remember the policy (favours exact-term matching). A real
# enterprise query log is a MIX of both, which is the point of the comparison.
QUESTIONS = [
    # Paraphrase queries
    ("How many days of annual leave can I carry into next year?", "leave-carry", "paraphrase"),
    ("What is the paid time off I earn each month?", "leave-annual", "paraphrase"),
    ("Do I need a doctor's note to take sick leave?", "leave-sick", "paraphrase"),
    ("How long is maternity leave here?", "leave-maternity", "paraphrase"),
    ("What's the hotel limit when I travel for work in a big city?", "expense-travel", "paraphrase"),
    ("By when must I file my expense claim before it is rejected?", "expense-claim", "paraphrase"),
    ("Can I claim meals without keeping the bills?", "expense-perdiem", "paraphrase"),
    ("Is it allowed to paste customer data into an outside AI tool?", "it-data", "paraphrase"),
    ("What approval do I need to buy something for 80 thousand rupees?", "procure-po", "paraphrase"),
    ("What happens before a new supplier can be paid?", "procure-vendor", "paraphrase"),
    ("Can I be let go during my first few months?", "probation", "paraphrase"),
    ("How many days a week can I work remotely?", "wfh-policy", "paraphrase"),
    ("Is there any money for home broadband when working from home?", "reimburse-internet", "paraphrase"),
    ("Where do I take a complaint if my manager does not resolve it?", "grievance", "paraphrase"),
    ("When do I have to hand back my laptop when leaving?", "asset-return", "paraphrase"),
    # Lexical queries (exact jargon, acronyms, figures)
    ("per diem meals incidentals domestic travel", "expense-perdiem", "lexical"),
    ("VPN multi-factor authentication remote access", "it-vpn", "lexical"),
    ("password 12 characters changed every 90 days", "it-password", "lexical"),
    ("referral bonus Rs 30000 six months", "referral-bonus", "lexical"),
    ("purchase order above Rs 50000 department head", "procure-po", "lexical"),
    ("home internet allowance Rs 1000 per month", "reimburse-internet", "lexical"),
    ("annual learning budget Rs 25000 certifications", "training-budget", "lexical"),
    ("carry forward 6 days lapses 31 March", "leave-carry", "lexical"),
    ("notice period 60 days managers resignation", "notice-period", "lexical"),
    ("maternity leave 26 weeks eligible", "leave-maternity", "lexical"),
    ("sick leave 12 days medical certificate", "leave-sick", "lexical"),
    ("probation 6 months 15 days notice", "probation", "lexical"),
    ("work from home 2 days per week manager approval", "wfh-policy", "lexical"),
    ("domestic travel economy airfare Rs 4000 per night hotel", "expense-travel", "lexical"),
    ("company assets laptop access card final settlement", "asset-return", "lexical"),
    # Identifier queries over the confusable form cluster: the code is the only
    # reliable disambiguator, which is exactly where exact-term matching wins.
    ("What is Form HR-21 used for?", "form-travel", "lexical"),
    ("Which form is IT-07?", "form-vpn", "lexical"),
    ("form for leave encashment approval", "form-leave", "lexical"),
    ("form to request a replacement laptop", "form-asset", "lexical"),
]

K = 3  # evaluate hit-rate within the top-K retrieved chunks


def tok(s):
    return s.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()


def rrf_fuse(dense_ranking, bm25_ranking, k=60):
    """Reciprocal Rank Fusion: each item scores sum of 1/(k + rank) across lists."""
    scores = {}
    for rank, idx in enumerate(dense_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def main():
    wl = WordLlama.load()
    ids = [c[0] for c in CHUNKS]
    texts = [c[1] for c in CHUNKS]

    # Dense index
    emb = np.array(wl.embed(texts))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # Lexical index
    bm25 = BM25Okapi([tok(t) for t in texts])

    strategies = ["dense", "bm25", "hybrid"]
    types = ["paraphrase", "lexical"]
    # counts[strategy][type] = (hits, rr_sum, n)
    stat = {s: {t: [0, 0.0, 0] for t in types + ["all"]} for s in strategies}

    per_q = []
    for q, gold, qtype in QUESTIONS:
        gold_idx = ids.index(gold)

        qv = np.array(wl.embed([q])[0])
        qv = qv / np.linalg.norm(qv)
        dense_rank = list(np.argsort(-(emb @ qv)))

        bm25_scores = bm25.get_scores(tok(q))
        bm25_rank = list(np.argsort(-bm25_scores))

        hybrid_rank = rrf_fuse(dense_rank, bm25_rank)

        ranks = {"dense": dense_rank, "bm25": bm25_rank, "hybrid": hybrid_rank}
        row = {"q": q, "gold": gold, "type": qtype}
        for s in strategies:
            pos = ranks[s].index(gold_idx)  # 0-based rank of the gold chunk
            hit = 1 if pos < K else 0
            for bucket in (qtype, "all"):
                stat[s][bucket][0] += hit
                stat[s][bucket][1] += 1.0 / (pos + 1)
                stat[s][bucket][2] += 1
            row[s] = pos + 1  # 1-based for readability
        per_q.append(row)

    n = len(QUESTIONS)
    print(f"Corpus: {len(CHUNKS)} chunks | Questions: {n} | hit-rate@{K}\n")
    header = f"{'strategy':10s}" + "".join(f"{t+' HR':>16s}" for t in types) + f"{'all HR':>10s}{'all MRR':>10s}"
    print(header)
    results = {}
    for s in strategies:
        results[s] = {}
        line = f"{s:10s}"
        for t in types:
            h, _, nn = stat[s][t]
            line += f"{h/nn:>16.2f}"
            results[s][t] = h / nn
        h, rrsum, nn = stat[s]["all"]
        line += f"{h/nn:>10.2f}{rrsum/nn:>10.3f}"
        results[s]["all_hit_rate"] = h / nn
        results[s]["all_mrr"] = rrsum / nn
        # robustness = the worse of the two query types (worst-case hit-rate)
        results[s]["worst_type"] = min(results[s][t] for t in types)
        print(line)

    print("\nRobustness (worst hit-rate across the two query types):")
    for s in strategies:
        print(f"  {s:10s} {results[s]['worst_type']:.2f}")

    with open("hybrid_results.json", "w") as f:
        json.dump({"k": K, "n": n, "results": results, "per_q": per_q}, f, indent=2)
    print("\nWrote hybrid_results.json")


if __name__ == "__main__":
    main()

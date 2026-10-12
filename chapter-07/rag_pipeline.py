"""The Chapter 7 pipeline in one module: chunking, a BM25 index, a dense index,
hybrid search with reciprocal rank fusion, and grounded answers with citations.

Imported by the Part B and Part C scripts, and reused by the Chapter 8 lab.
The code follows the listings in Section 7.7.
"""
import json, re, pathlib
import numpy as np
from rank_bm25 import BM25Okapi          # pip install rank_bm25
from wordllama import WordLlama          # pip install wordllama

HERE = pathlib.Path(__file__).parent
# Common English words, saved from spaCy's list so that no spaCy install is needed.
STOP = set((HERE / "stopwords.txt").read_text(encoding="utf-8").split())

def tokens(text):                        # keep codes such as E-214 and RC-OF-118 whole
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if w not in STOP]

def chunk(doc, size=50, overlap=10, title=True):
    words, step = doc["text"].split(), max(1, size - overlap)
    out = []
    for s in range(0, max(1, len(words) - overlap), step):
        prefix = doc["title"] + ". " if title else ""
        out.append({"doc": doc["id"], "text": prefix + " ".join(words[s:s + size])})
        if s + size >= len(words):
            break
    return out

kb = json.load(open(HERE / "nilgiri_kb.json", encoding="utf-8"))   # 30 documents, 36 questions
chunks = [c for d in kb["docs"] for c in chunk(d)]
bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])               # keyword index
wl = WordLlama.load()                                               # small embedding model
E = np.asarray(wl.embed([c["text"] for c in chunks]), dtype=np.float32)
E = E / np.linalg.norm(E, axis=1, keepdims=True)                    # dense index

def hybrid_search(question, k=60):
    q = np.asarray(wl.embed([question])[0], dtype=np.float32); q = q / np.linalg.norm(q)
    rankings = [np.argsort(-bm25.get_scores(tokens(question))), np.argsort(-(E @ q))]
    rrf = np.zeros(len(chunks))
    for ranking in rankings:                                        # reciprocal rank fusion
        for position, i in enumerate(ranking):
            rrf[i] += 1 / (k + position + 1)
    return [chunks[i] for i in np.argsort(-rrf)]

SYSTEM = """You are a service assistant for Nilgiri Flow Systems field engineers.
Answer ONLY from the numbered sources provided. Cite sources in square brackets, like [2].
If the sources do not contain the answer, reply exactly: "I could not find this in the
service documents." Never invent part numbers, alarm codes or values."""

def build_messages(question, hits):
    sources = "\n".join(f'<source id="{i + 1}" doc="{h["doc"]}">{h["text"]}</source>'
                        for i, h in enumerate(hits))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{sources}\n\nQuestion: {question}"}]

def answer(question, call_model, model=None, top_n=4):
    hits = hybrid_search(question)[:top_n]
    reply = call_model(build_messages(question, hits), model=model, temperature=0, max_tokens=300)
    return reply["text"], hits

def offline_stub(messages, model=None, temperature=0, max_tokens=300):
    """Stands in for a model so the pipeline runs without a key. It picks the source that
    shares most words with the question and quotes it with a citation. It cannot reason
    or ask a clarifying question, so use it only to check the plumbing."""
    user = messages[-1]["content"]
    qwords = set(tokens(user.rsplit("Question:", 1)[-1]))
    sources = re.findall(r'<source id="(\d+)"[^>]*>(.*?)</source>', user, re.S)
    overlap, sid, body = max((len(qwords & set(tokens(b))), -int(i), b) for i, b in sources)
    if overlap < 2:
        text = "I could not find this in the service documents."
    else:
        text = " ".join(body.split()[:40]) + f"... [{-sid}]"
    return {"text": text, "input_tokens": len(user) // 4, "output_tokens": len(text) // 4}

def get_backend(name):
    """'stub', 'anthropic' or 'openai' (the last also covers local servers, Chapter 8)."""
    if name == "stub":
        return offline_stub
    import sys
    sys.path.insert(0, str(HERE.parent / "chapter-05"))
    import providers                     # the adapters from Chapter 5
    return {"anthropic": providers.call_anthropic,
            "openai": providers.call_openai_compatible}[name]

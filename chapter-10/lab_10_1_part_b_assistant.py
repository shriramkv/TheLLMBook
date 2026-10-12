"""Lab 10.1, Part B (Section 10.6): an assistant that sees, listens and speaks.

Late fusion (Figure 10.4): the voice note becomes text with Whisper, the photo becomes text
with a vision model, the Chapter 7 hybrid search finds the manual page, a language model
writes a short spoken-style answer, and Coqui TTS reads it out.

    pip install openai openai-whisper coqui-tts rank_bm25 wordllama
    # system packages: ffmpeg (Whisper) and espeak-ng (Coqui TTS)
    export OPENAI_API_KEY=...          # or OPENAI_BASE_URL for a local server (Chapter 8)
    export VISION_MODEL=gpt-4o-mini TEXT_MODEL=gpt-4o-mini
    python lab_10_1_part_b_assistant.py
    python lab_10_1_part_b_assistant.py --offline     # no model calls: checks the plumbing

Any step whose software or model is missing falls back to a clearly labelled stand-in, so
you can build the pipeline one step at a time. The script prints the time for each step.
"""
import argparse, base64, logging, os, re, sys, time, json, pathlib
import numpy as np
from rank_bm25 import BM25Okapi
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "chapter-07"))
from rag_pipeline import tokens, chunk, wl, offline_stub      # the Chapter 7 building blocks

p = argparse.ArgumentParser()
p.add_argument("--audio", default="samples/voice_note.wav")
p.add_argument("--photo", default="samples/belt.jpg")
p.add_argument("--whisper-size", default="small")
p.add_argument("--offline", action="store_true")
args = p.parse_args()
timings = {}

def timed(name):
    def wrap(fn):
        def run(*a, **k):
            t0 = time.perf_counter(); out = fn(*a, **k)
            timings[name] = time.perf_counter() - t0
            return out
        return run
    return wrap

client = None
if not args.offline:
    from openai import OpenAI
    client = OpenAI()
VISION_MODEL = os.getenv("VISION_MODEL")
TEXT_MODEL = os.getenv("TEXT_MODEL")

# ---- Step 1: Listen
@timed("1 listen (Whisper)")
def listen(path):
    try:
        import whisper
        asr = whisper.load_model(args.whisper_size)
        return asr.transcribe(path)["text"].strip()             # language is detected automatically
    except Exception as e:
        print(f"[stand-in] Whisper unavailable ({type(e).__name__}); using samples/voice_note.txt")
        return open("samples/voice_note.txt", encoding="utf-8").read().strip()

# ---- Step 2: See
def describe_photo(client, model, path, question):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    msg = [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    reply = client.chat.completions.create(model=model, messages=msg, max_tokens=200)
    return reply.choices[0].message.content

@timed("2 see (vision model)")
def see(path):
    if client is None or not VISION_MODEL:
        print("[stand-in] no vision model set; using a fixed description")
        return "Two V-belts on a pulley drive. The lower belt run shows cracks across the ribs, and the upper belt looks glazed."
    return describe_photo(client, VISION_MODEL, path,
                          "Describe the condition of the equipment in this photo for a maintenance technician. "
                          "Say only what is visible. If something is unclear, say so.")

# ---- Step 3: Think (retrieve, then answer in a form meant to be heard)
manual = json.load(open("field_manual.json", encoding="utf-8"))
chunks = [c for d in manual["docs"] for c in chunk(d)]
bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
E = np.asarray(wl.embed([c["text"] for c in chunks]), dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)

def hybrid_search(question, k=60):
    q = np.asarray(wl.embed([question])[0], dtype=np.float32); q /= np.linalg.norm(q)
    rrf = np.zeros(len(chunks))
    for ranking in (np.argsort(-bm25.get_scores(tokens(question))), np.argsort(-(E @ q))):
        for pos, i in enumerate(ranking):
            rrf[i] += 1 / (k + pos + 1)
    return [chunks[i] for i in np.argsort(-rrf)]

SPOKEN = """You are a maintenance assistant speaking to a technician whose hands are busy.
Answer ONLY from the numbered manual sources. Reply in two or three short sentences meant to be
heard, not read: the most important action first, no lists, no symbols, no source numbers.
If the sources do not cover the question, say so and suggest calling the supervisor."""

@timed("3 think (retrieve + model)")
def think(transcript, description):
    question = f"Technician's voice note: {transcript}\nWhat the photo shows: {description}"
    hits = hybrid_search(transcript + " " + description)[:4]
    sources = "\n".join(f'<source id="{i + 1}" doc="{h["doc"]}">{h["text"]}</source>' for i, h in enumerate(hits))
    messages = [{"role": "system", "content": SPOKEN},
                {"role": "user", "content": f"{sources}\n\nQuestion: {question}"}]
    if client is None or not TEXT_MODEL:
        print("[stand-in] no text model set; quoting the best manual passage")
        return offline_stub(messages)["text"], hits
    r = client.chat.completions.create(model=TEXT_MODEL, messages=messages, max_tokens=150, temperature=0)
    return r.choices[0].message.content, hits

# ---- Step 4: Speak
@timed("4 speak (Coqui TTS)")
def speak(text, path="reply.wav"):
    try:
        logging.disable(logging.INFO)                           # Coqui TTS is chatty
        from TTS.api import TTS
        tts = TTS("tts_models/en/vctk/vits")
        speaker = next(s for s in tts.speakers if s.strip() == "p248")   # an Indian English voice
        tts.tts_to_file(text=text, speaker=speaker, file_path=path)
        return path
    except Exception as e:
        print(f"[stand-in] TTS unavailable ({type(e).__name__}: {e}); reply not spoken")
        return None

transcript = listen(args.audio)
print("\nHeard:", transcript)
description = see(args.photo)
print("\nSaw:", description)
reply, hits = think(transcript, description)
print("\nManual pages used:", ", ".join(dict.fromkeys(h["doc"] for h in hits)))
print("\nReply:", reply)
wav = speak(re.sub(r"\s*\[\d+\]", "", reply))     # never read citation marks aloud
if wav:
    print("\nSpoken reply saved to", wav)
print("\nTime per step:")
for k, v in timings.items():
    print(f"  {k:28s} {v:6.2f} s")

"""Lab 7.1, Part B (Section 7.7): grounded answers with citations.

    python lab_7_1_part_b_generate.py stub                       # offline, checks the plumbing
    python lab_7_1_part_b_generate.py anthropic claude-sonnet-4-5  # needs ANTHROPIC_API_KEY
    python lab_7_1_part_b_generate.py openai gpt-4o-mini           # needs OPENAI_API_KEY
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=x \
        python lab_7_1_part_b_generate.py openai llama3.1:8b       # a local Ollama server (Chapter 8)

Use whichever current model names your provider lists. Add your own questions at the end of
the command line in quotes.
"""
import sys
from rag_pipeline import answer, get_backend

backend = sys.argv[1] if len(sys.argv) > 1 else "stub"
model = sys.argv[2] if len(sys.argv) > 2 else None
questions = sys.argv[3:] or [
    "Which filter fits an RC-75?",                      # ambiguous: three different filters
    "What does alarm E-214 mean on the compressor?",
    "The pump sounds like it has stones rattling inside. What is happening?",
    "What is the part number of the NF-200 impeller nut?",   # not in the knowledge base
]
call_model = get_backend(backend)
if backend == "stub":
    print("Offline stub: answers are copied sentences, not model output.\n")

for q in questions:
    text, hits = answer(q, call_model, model=model)
    print("Q:", q)
    print("Sources:", ", ".join(f"[{i + 1}] {h['doc']}" for i, h in enumerate(hits)))
    print("A:", text, "\n")

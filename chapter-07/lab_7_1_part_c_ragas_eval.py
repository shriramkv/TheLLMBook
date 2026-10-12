"""Lab 7.1, Part C (Section 7.7): score generated answers with RAGAS.

RAGAS uses a model as a judge, so this part needs an API key. It was written against
ragas 0.2 and 0.3; field names and imports change between versions, so if an import fails,
check the RAGAS documentation for the version you have installed.

    pip install "ragas>=0.2,<0.4" langchain-openai
    export OPENAI_API_KEY=...
    python lab_7_1_part_c_ragas_eval.py openai gpt-4o-mini

The first two arguments choose the model that answers (as in Part B). The judge is set up
below with OpenAI through LangChain; swap in another provider by changing judge_llm and
judge_embeddings. `python lab_7_1_part_c_ragas_eval.py stub` builds the records offline
and stops before scoring, which is useful for checking the data.
"""
import sys, json
from rag_pipeline import answer, get_backend

backend = sys.argv[1] if len(sys.argv) > 1 else "stub"
model = sys.argv[2] if len(sys.argv) > 2 else None
call_model = get_backend(backend)

test_cases = [json.loads(line) for line in open("test_cases.jsonl", encoding="utf-8")]
records = []
for case in test_cases:                    # each has a question and a reference answer
    text, hits = answer(case["question"], call_model, model=model)
    records.append({"user_input": case["question"],
                    "retrieved_contexts": [h["text"] for h in hits],
                    "response": text,
                    "reference": case["reference"]})
json.dump(records, open("ragas_records.json", "w"), indent=1)
print(f"Built {len(records)} records and saved them to ragas_records.json")
if backend == "stub":
    print("Stub answers are not worth judging; rerun with a real model to score them.")
    sys.exit()

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

result = evaluate(dataset=EvaluationDataset.from_list(records),
                  metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
                  llm=judge_llm, embeddings=judge_embeddings)
print(result)                              # one average score per metric
df = result.to_pandas()
df.to_csv("ragas_scores.csv", index=False)
print("\nLowest faithfulness first (inspect these by hand):")
print(df.sort_values("faithfulness")[["user_input", "faithfulness"]].head(5).to_string(index=False))

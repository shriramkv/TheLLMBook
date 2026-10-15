# Generative AI and Large Language Models: companion code

Code for Chapters 1 to 16 of *Generative AI and Large Language Models: Concepts, Architectures and Hands-on Applications*
(Dr. Shriram K. Vasudevan, Arun G K, Subashri Vasudevan).

## Setup
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Contents
| Folder | File | Book section | Needs |
|---|---|---|---|
| chapter-01 | lab-1-1-meet-the-models.ipynb (and .py) | Lab 1.1 | transformers, torch, internet |
| chapter-02 | sec_2_1_tensor_operations.py | 2.1 | numpy |
| chapter-02 | sec_2_2_softmax_entropy.py | 2.2 | numpy |
| chapter-02 | sec_2_3_embeddings.py | 2.3 | numpy |
| chapter-02 | sec_2_4_autograd.py | 2.4 | torch |
| chapter-02 | sec_2_6_overfitting_experiment.py | 2.6, Figure 2.5 | numpy |
| chapter-02 | lab-2-1-tiny-network.ipynb (and .py) | Lab 2.1 (Section 2.7) | numpy |
| chapter-03 | sec_3_2_self_attention.py | 3.2 | numpy |
| chapter-03 | sec_3_3_positional_encoding.py | 3.3 | numpy |
| chapter-03 | sec_3_4_causal_mask.py | 3.4 | numpy |
| chapter-03 | download_data.py | 3.6 | internet |
| chapter-03 | mini_gpt.py | 3.6, Lab 3.1 | torch (about 7 min on a 2-core CPU) |
| chapter-03 | visualise_attention.py | 3.7, Figure 3.4 | torch, matplotlib; run mini_gpt.py first |
| chapter-04 | lab_4_1_part_a_bpe.py | 4.1, Lab 4.1 Part A | input.txt from chapter-03 |
| chapter-04 | lab_4_1_part_b_tokeniser_languages.py | 4.1, Figure 4.1, Lab 4.1 Part B | tiktoken, internet on first run |
| chapter-04 | lab_4_1_part_c_kv_cache.py | 4.4, Lab 4.1 Part C | torch; copy mini_gpt.py and minigpt.pt from chapter-03 |
| chapter-05 | structured.py | 5.4 | pydantic |
| chapter-05 | load_prompt.py, prompts/*.yaml | 5.5 | pyyaml, jinja2 |
| chapter-05 | providers.py, workbench.py, claims_eval.jsonl | 5.8, Lab 5.1 | run `python workbench.py stub` offline, or `anthropic` / `openai` with an API key |
| chapter-06 | lora_experiment.py, data/licences_*.txt | 6.3, 6.7, Lab 6.1 Part A, Table 6.7, Figure 6.4 | torch; copy mini_gpt.py, minigpt.pt and input.txt from chapter-03 (about 12 min on a CPU) |
| chapter-06 | sec_6_2_memory_estimator.py | 6.2, Table 6.4 | none |
| chapter-06 | make_sample_data.py, data/train.jsonl, data/valid.jsonl | 6.4 | none |
| chapter-06 | lab_6_1_part_b_qlora.py | 6.5, Lab 6.1 Part B | GPU with 16 GB or more; transformers, peft, trl, datasets, bitsandbytes, accelerate |
| chapter-07 | make_corpus.py, nilgiri_kb.json | 7.7 | none (the knowledge base is already included) |
| chapter-07 | rag_pipeline.py, stopwords.txt | 7.3, 7.4, 7.7 | rank_bm25, wordllama; shared by Parts B and C and by chapters 8 and 10 |
| chapter-07 | lab_7_1_part_a_retrieval.py | 7.4, Lab 7.1 Part A, Tables 7.5 and 7.6, Figure 7.2 | rank_bm25, wordllama (under a minute) |
| chapter-07 | lab_7_1_part_b_generate.py | Lab 7.1 Part B | run with `stub` offline, or `anthropic` / `openai` with a key; uses chapter-05/providers.py |
| chapter-07 | lab_7_1_part_c_ragas_eval.py, test_cases.jsonl | 7.6, Lab 7.1 Part C | ragas, langchain-openai, an API key for the judge |
| chapter-08 | sec_8_1_transformers_pipeline.py | 8.1 | transformers, accelerate, internet on first run |
| chapter-08 | sec_8_2_frameworks.py | 8.2 | langchain, langchain-openai, llama-index, an API key |
| chapter-08 | sec_8_3_call_with_retries.py | 8.3 | none (simulated failures) |
| chapter-08 | sec_8_4_one_client.py | 8.4 | openai; Ollama or vLLM running |
| chapter-08 | sec_8_5_gpu_estimator.py | 8.5, Figure 8.2, Table 8.4 | none; matplotlib for `--plot` |
| chapter-08 | backends.py, lab_8_1_four_modes.py, not_in_kb.txt | 8.6, Lab 8.1, Table 8.6 | run with `stub` offline; then api, inprocess, ollama, vllm |
| chapter-09 | sec_9_2_agent_loop.py | 9.2 | none offline (scripted model); openai for a live model |
| chapter-09 | sec_9_3_mcp_po_server.py, sec_9_3_mcp_client_test.py | 9.3 | mcp (Python SDK version 2) |
| chapter-09 | sec_9_4_langgraph_workflow.py | 9.4, Lab 9.1 exercise 4 | langgraph |
| chapter-09 | invoice_workflow.py | 9.8, Lab 9.1, Table 9.6 | none |
| chapter-10 | tiny_diffusion.py | 10.1, Lab 10.1 Part A, Table 10.5, Figure 10.2 | torch (about 15 s on a CPU); matplotlib for `--plot` |
| chapter-10 | lab_10_1_part_b_assistant.py, field_manual.json, samples/ | 10.3, 10.5, Lab 10.1 Part B | openai-whisper, coqui-tts, openai, rank_bm25, wordllama; ffmpeg and espeak-ng; `--offline` to test |
| chapter-10 | lab_10_1_part_c_job_card.py, samples/job_card.jpg | 10.2, Lab 10.1 Part C | pydantic, openai and a vision model; `--offline` to test |
| chapter-10 | make_samples.py | 10.6 | Pillow, coqui-tts (the samples are already included) |
| chapter-11 | semantic_cache_experiment.py, cache_pairs.json | 11.4, semantic cache limits | wordllama |
| chapter-11 | quantise_minigpt.py | 11.2, quantisation loss | torch; mini_gpt.py, minigpt.pt and input.txt included in the folder |
| chapter-11 | cost_model.py | 11.5, cost waterfall | none |
| chapter-11 | lab/llm_service.py, lab/config.yaml | 11 Lab, serving | fastapi, uvicorn, pyyaml |
| chapter-11 | lab/loadtest.py | 11 Lab, load test | httpx; run the service first |
| chapter-11 | eval_gate.py | 11, release gate | none (exits non-zero to block a ship) |
| chapter-12 | defences_experiment.py | 12, measuring defences | none |
| chapter-12 | courier_app.py | 12, vulnerable vs hardened agent | none |
| chapter-12 | run_lab.py | 12, Lab | none |
| chapter-13 | weat_experiment.py | 13.1, Figure 13.1, WEAT bias test | wordllama, numpy |
| chapter-14 | roi_model.py | 14.2, Figure 14.1, ROI and tornado sensitivity | numpy |
| chapter-15 | hybrid_rag_experiment.py | 15.1, Figure 15.1, dense vs lexical vs hybrid retrieval | wordllama, rank_bm25, numpy |
| chapter-16 | test_time_compute.py | 16.2, Figure 16.1, self-consistency / test-time compute | none |

Run each script from inside its own folder, for example:

    cd chapter-03
    python download_data.py
    python mini_gpt.py
    python visualise_attention.py

## Notes for Chapters 6 to 10

- **Offline first.** Every lab has a way to run without an API key or GPU (`stub`, `--offline`, or a scripted model), so you can check the plumbing before spending anything. Stand-in outputs are labelled as such and say nothing about model quality.
- **Model names.** Scripts that call a model take its name from the command line or an environment variable (`API_MODEL`, `OLLAMA_MODEL`, `VLLM_MODEL`, `LOCAL_HF_MODEL`, `VISION_MODEL`, `TEXT_MODEL`). Use whichever current models your provider lists.
- **Shared code.** Chapters 8 and 10 import the Chapter 7 pipeline from `../chapter-07`, and Chapter 7 imports the provider adapters from `../chapter-05`, so keep the folders side by side.
- **Chapter 9** needs no model at all for the main lab: the agents are rule-based stand-ins, and the exercises replace them with model calls.
- **Chapter 10 samples** (`belt.jpg`, `job_card.jpg`, `voice_note.wav`) are synthetic, made with `make_samples.py`. Replace them with your own photographs and recordings to judge accuracy.
- **System packages.** Whisper needs `ffmpeg`; Coqui TTS needs `espeak-ng`. On Ubuntu: `sudo apt install ffmpeg espeak-ng`; on macOS: `brew install ffmpeg espeak-ng`.
- Libraries for fine-tuning, agents and evaluation change quickly. If an import fails, check the documentation for the version you installed; the comments in each script say which version it was written against.

## Notes for Chapters 11 to 16

- **Fully offline and reproducible.** Every experiment in these chapters runs with no API keys and no network at run time, and each one produces the exact figures and tables in the book on any machine. Each chapter folder also has its own short `README.md`.
- **Reference outputs included.** Each folder ships the `*_results.json` it produces (for example `quant_results.json`, `weat_results.json`, `roi_results.json`, `hybrid_results.json`, `ttc_results.json`), so you can diff your run against the book's.
- **Chapter 11** reuses the Chapter 3 MiniGPT; `mini_gpt.py`, `minigpt.pt` and `input.txt` are copied into `chapter-11/` so the quantisation experiment runs on its own. The lab service (`lab/`) is a FastAPI, OpenAI-compatible server with a cache, rate limiter and metrics; run it with `uvicorn lab.llm_service:app` and drive it with `lab/loadtest.py`.
- **Chapter 12** measures defences, not attacks: the attempts are generated from templates at the pattern level only to score how well each defence holds. The lesson is that guardrails belong in code, as `courier_app.py` shows with its hardened build.
- **Chapters 13 and 15** reuse the Chapter 7 WordLlama embeddings; **Chapters 14 and 16** need only NumPy or the standard library.

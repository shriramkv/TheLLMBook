# The LLM Book, Companion Code: Chapters 11 to 16

Companion code for *Generative AI and Large Language Models: Concepts, Architectures and Hands-on Applications* (McGraw-Hill), by Dr. Shriram K. Vasudevan, Arun G K and Subashri Vasudevan.

This folder covers **Chapters 11 to 16**. Each chapter has its own folder with the original experiment behind that chapter's figure, plus a short `README.md`. Every experiment is **offline and reproducible**: no API keys, and no network access at run time. The numbers in the book regenerate exactly on any machine.

## What is here

| Folder | Chapter | The experiment it runs |
| --- | --- | --- |
| `chapter-11/` | LLMOps: Deploying GenAI to Production | Semantic-cache limits, MiniGPT quantisation, a FastAPI LLM service with a load test, a cost model, and a release gate |
| `chapter-12/` | Security, Safety and Guardrails | Measuring defences against templated attack attempts; a vulnerable vs hardened agent app |
| `chapter-13/` | Ethics, Governance and Regulation | A WEAT bias test on sentence embeddings |
| `chapter-14/` | GenAI in the Enterprise | An ROI model with a tornado sensitivity analysis |
| `chapter-15/` | Capstone Projects | Dense vs lexical vs hybrid retrieval, measured by query type |
| `chapter-16/` | The Future of Generative AI | The exact self-consistency / test-time-compute curve |

Each chapter folder also ships the `*_results.json` it produces, so you can diff your run against the book's.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later is recommended. Some experiments (the retrieval and bias ones) use `wordllama` and `rank_bm25`; the quantisation experiment uses `torch`. The ethics, ROI and test-time-compute experiments need only NumPy or the standard library. See each chapter's `README.md` for exact commands.

## A note on the code

These are teaching experiments, written to be read. They favour clarity over cleverness, run on a laptop, and measure something you can check. The point of each one is a method you can reuse on your own data, not the specific numbers it prints.

## Links

- Main repository: https://github.com/shriramkv/TheLLMBook
- Author: Dr. Shriram K. Vasudevan (https://github.com/shriramkv)

# Companion code for Generative AI and Large Language Models

Code for the sample chapters of *Generative AI and Large Language Models: Concepts, Architectures and Hands-on Applications*
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

Run each script from inside its own folder, for example:

    cd chapter-03
    python download_data.py
    python mini_gpt.py
    python visualise_attention.py

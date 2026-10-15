# Chapter 12: Security, Safety and Guardrails

Experiments that measure **defences**, not attacks. The attack attempts are generated from templates at the pattern level purely to score how well each defence holds; there is no usable attack recipe here, by design.

## Files

- `defences_experiment.py` — generates templated attempt variants against an over-obedient model and measures how far each defence layer (none, prompt-only, input filter, code checks) reduces success. Standard library only.
- `courier_app.py` — a `VulnerableCourier` and a `HardenedCourier` side by side. The hardened version checks ownership and eligibility **in code**, not in the prompt, so a crafted request cannot talk it into acting outside its authority.
- `run_lab.py` — runs four goals (including an indirect-injection attempt) against both builds and reports the success rate of each.

## Run

```bash
python3 defences_experiment.py
python3 run_lab.py
```

`defence_results.json` and `lab_results.json` hold the book's runs: defences drop attempt success from 100% (none) to 0% (code checks), and the hardened agent resists everything the vulnerable one falls to.

The lesson: guardrails belong in code. A prompt instruction to "only act on the user's own data" can be argued away; a function that checks ownership before acting cannot.

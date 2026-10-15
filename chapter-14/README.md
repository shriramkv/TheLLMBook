# Chapter 14: GenAI in the Enterprise

One experiment: a transparent business-case model with a sensitivity analysis.

## Files

- `roi_model.py` — models the economics of automating part of a support workload: baseline cost, new cost, annual saving, payback, and a tornado sensitivity analysis that swings each assumption and ranks its effect. Standard library plus NumPy.

## Run

```bash
python3 roi_model.py
```

`roi_results.json` holds the book's run: a 3.74 crore annual saving and a 4.8-month payback in the base case. The tornado makes the real point: the business case is decided by the deflection rate, the ticket volume and the cost of the human work, while the per-ticket model cost that teams obsess over barely moves the result.

Change the assumptions in the `Case` dataclass to model your own use case, and read the tornado before deciding which lever to pull.

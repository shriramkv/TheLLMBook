"""
roi_model.py  -  A reproducible enterprise GenAI business-case model.

Chapter 14, "GenAI in the Enterprise". No external services, no API keys,
pure arithmetic, so the numbers reproduce exactly on any machine. The point
is not the specific rupee figures (they are illustrative) but the METHOD:
a transparent model you can argue about, and a sensitivity analysis that
shows which assumption actually decides the outcome.

Scenario: a mid-size Indian enterprise automates part of its customer-support
workload with a RAG assistant (Chapter 7), served through the LLMOps stack of
Chapter 11, governed by the process of Chapter 13.
"""
from dataclasses import dataclass, asdict, replace


@dataclass
class Case:
    # Volume and the status quo
    tickets_per_year: float = 2_000_000
    human_cost_per_ticket: float = 60.0      # Rs, blended cost of one agent interaction

    # What the assistant changes
    deflection_rate: float = 0.45            # fraction handled end-to-end by the assistant
    rework_rate: float = 0.08                # fraction of deflected tickets that bounce back to a human
    review_rate: float = 0.05                # fraction of deflected tickets a human QA-reviews
    review_cost_per_ticket: float = 60.0     # Rs, a review costs about one human touch
    ai_cost_per_ticket: float = 4.0          # Rs, model + infra per ticket (see Chapter 11)

    # What it costs to build and run
    build_cost: float = 15_000_000           # Rs, one-off: integration, RAG, eval harness, governance
    run_cost_per_year: float = 6_000_000     # Rs/year: platform, monitoring, the people who own it


def economics(c: Case):
    baseline = c.tickets_per_year * c.human_cost_per_ticket

    ai_tickets = c.tickets_per_year * c.deflection_rate
    human_tickets = c.tickets_per_year * (1 - c.deflection_rate)

    human_cost = human_tickets * c.human_cost_per_ticket
    ai_cost = ai_tickets * c.ai_cost_per_ticket
    review_cost = ai_tickets * c.review_rate * c.review_cost_per_ticket
    rework_cost = ai_tickets * c.rework_rate * c.human_cost_per_ticket

    new_annual = human_cost + ai_cost + review_cost + rework_cost + c.run_cost_per_year
    annual_saving = baseline - new_annual

    payback_months = (c.build_cost / (annual_saving / 12)) if annual_saving > 0 else float("inf")
    # First-year net counts the one-off build cost; steady state does not.
    first_year_net = annual_saving - c.build_cost
    return {
        "baseline": baseline,
        "new_annual": new_annual,
        "annual_saving": annual_saving,
        "payback_months": payback_months,
        "first_year_net": first_year_net,
        "breakdown": {
            "human (residual)": human_cost,
            "AI inference": ai_cost,
            "human review (QA)": review_cost,
            "rework (bounced)": rework_cost,
            "platform + run": c.run_cost_per_year,
        },
    }


def tornado(base: Case, swing: float = 0.30):
    """One-at-a-time sensitivity: swing each driver +/- `swing` and record the
    change in annual saving. The widest bars are the assumptions that decide
    the business case; the narrow ones are the ones teams argue about for no
    reason."""
    base_saving = economics(base)["annual_saving"]
    drivers = [
        "deflection_rate", "human_cost_per_ticket", "rework_rate",
        "review_rate", "ai_cost_per_ticket", "run_cost_per_year",
        "tickets_per_year",
    ]
    rows = []
    for d in drivers:
        v = getattr(base, d)
        lo = economics(replace(base, **{d: v * (1 - swing)}))["annual_saving"]
        hi = economics(replace(base, **{d: v * (1 + swing)}))["annual_saving"]
        rows.append((d, lo - base_saving, hi - base_saving))
    # widest total span first
    rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)
    return base_saving, rows


def rupees(x):
    # Indian grouping, in crore for readability
    cr = x / 1e7
    return f"Rs {x:,.0f}  (Rs {cr:.2f} crore)"


if __name__ == "__main__":
    base = Case()
    e = economics(base)
    print("=== Base case: support automation ===")
    print("Assumptions:")
    for k, v in asdict(base).items():
        print(f"  {k:24s} {v:,.2f}")
    print()
    print("Baseline annual cost  :", rupees(e["baseline"]))
    print("New annual cost       :", rupees(e["new_annual"]))
    print("Annual saving         :", rupees(e["annual_saving"]))
    print(f"Payback period        : {e['payback_months']:.1f} months")
    print("First-year net        :", rupees(e["first_year_net"]))
    print()
    print("New-cost breakdown:")
    for k, v in e["breakdown"].items():
        print(f"  {k:20s} {rupees(v)}")
    print()

    base_saving, rows = tornado(base, swing=0.30)
    print("=== Sensitivity (annual saving, +/- 30% each driver) ===")
    print(f"Base annual saving: {rupees(base_saving)}")
    print(f"{'driver':24s} {'-30%':>16s} {'+30%':>16s} {'span':>16s}")
    for d, lo, hi in rows:
        print(f"{d:24s} {lo/1e6:>14.2f}M {hi/1e6:>14.2f}M {abs(hi-lo)/1e6:>14.2f}M")

    # Break-even deflection: the deflection rate at which annual saving = 0
    import numpy as np
    for dr in np.linspace(0.0, 0.45, 46):
        if economics(replace(base, deflection_rate=float(dr)))["annual_saving"] > 0:
            print(f"\nBreak-even deflection rate: ~{dr:.2f} "
                  f"(below this, automation loses money at these assumptions)")
            break

    # Emit chart data for the figure
    import json
    chart = {
        "base_saving": base_saving,
        "tornado": [{"driver": d, "lo": lo, "hi": hi} for d, lo, hi in rows],
        "breakdown": e["breakdown"],
        "payback_months": e["payback_months"],
    }
    with open("roi_results.json", "w") as f:
        json.dump(chart, f, indent=2)
    print("\nWrote roi_results.json")

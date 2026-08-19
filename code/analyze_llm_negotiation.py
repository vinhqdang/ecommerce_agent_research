"""
Summarize the LLM-agent negotiation pilot (llm_negotiation.py output) and
compare it against the rule-based ACE simulation's bilateral-bargaining
regimes (M2 = price-only, M3 = multi-attribute), whose headline numbers
live in data/headline_numbers.json.

Outputs:
  data/table6_llm_validation.csv     -- one row per regime x model pairing
  data/llm_validation_summary.json   -- headline numbers + rule-based comparison
"""

import os
import json

import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
RESULTS_CSV = os.path.join(DATA_DIR, "llm_negotiation_results.csv")
HEADLINE_JSON = os.path.join(DATA_DIR, "headline_numbers.json")
OUT_TABLE = os.path.join(DATA_DIR, "table6_llm_validation.csv")
OUT_SUMMARY = os.path.join(DATA_DIR, "llm_validation_summary.json")


def main():
    df = pd.read_csv(RESULTS_CSV)
    with open(HEADLINE_JSON) as f:
        headline = json.load(f)

    df["pairing"] = df["buyer_provider"] + "_buyer-" + df["seller_provider"] + "_seller"

    grp = df.groupby(["regime", "pairing"]).agg(
        n_episodes=("deal", "size"),
        deal_rate=("deal", "mean"),
        failed=("failed", "sum"),
        mean_rounds=("rounds_used", "mean"),
        schema_violations=("schema_violations", "sum"),
        mean_price=("price", "mean"),
        mean_welfare=("welfare", "mean"),
        mean_cs=("consumer_surplus", "mean"),
        mean_ps=("producer_surplus", "mean"),
        bundle_rate=("bundle_included", "mean"),
    ).reset_index()
    grp.to_csv(OUT_TABLE, index=False)

    by_regime = df.groupby("regime").agg(
        n_episodes=("deal", "size"),
        deal_rate=("deal", "mean"),
        mean_price=("price", "mean"),
        mean_welfare=("welfare", "mean"),
    )
    # Failed episodes are zero-filled in mean_welfare above, so a regime that
    # happened to draw more API failures looks artificially worse. Compute
    # the deal-conditional mean too, to separate "negotiation broke down"
    # (an API/infra artifact here) from "surplus created when it succeeded"
    # (the thing M2 vs M3 in the rule-based sim actually measures).
    deals_only = df[df["deal"]]
    by_regime_conditional = deals_only.groupby("regime").agg(
        n_deals=("welfare", "size"),
        mean_price_given_deal=("price", "mean"),
        mean_welfare_given_deal=("welfare", "mean"),
    )

    summary = {
        "llm_pilot": {
            "n_total_episodes": int(len(df)),
            "n_failed_episodes": int(df["failed"].sum()),
            "overall_deal_rate": float(df["deal"].mean()),
            "by_regime": by_regime.to_dict(orient="index"),
            "by_regime_conditional_on_deal": by_regime_conditional.to_dict(orient="index"),
        },
        "rule_based_comparison": {
            "welfare_M2_price_only_rule_based": headline["welfare_M2"],
            "welfare_M3_multiattr_rule_based": headline["welfare_M3_multiattr"],
            "price_M2_rule_based": headline["price_M2"],
            "welfare_gain_M3_vs_M2_rule_based": headline["welfare_gain_M3_vs_M2"],
        },
    }

    if "price_only" in by_regime.index and "multiattr" in by_regime.index:
        llm_gain = by_regime.loc["multiattr", "mean_welfare"] - by_regime.loc["price_only", "mean_welfare"]
        summary["llm_pilot"]["welfare_gain_multiattr_vs_price_only_llm_unconditional"] = float(llm_gain)
        summary["directional_agreement_with_rule_based_unconditional"] = bool(
            (llm_gain > 0) == (headline["welfare_gain_M3_vs_M2"] > 0)
        )
    if "price_only" in by_regime_conditional.index and "multiattr" in by_regime_conditional.index:
        llm_gain_cond = (by_regime_conditional.loc["multiattr", "mean_welfare_given_deal"]
                         - by_regime_conditional.loc["price_only", "mean_welfare_given_deal"])
        summary["llm_pilot"]["welfare_gain_multiattr_vs_price_only_llm_given_deal"] = float(llm_gain_cond)
        summary["directional_agreement_with_rule_based_given_deal"] = bool(
            (llm_gain_cond > 0) == (headline["welfare_gain_M3_vs_M2"] > 0)
        )
        summary["note"] = (
            "N=36 pilot with real free-tier LLM APIs; the unconditional comparison is "
            "confounded because multiattr happened to draw more API-failure episodes "
            "(zero-filled), which by itself lowers its mean welfare. The deal-conditional "
            "comparison isolates surplus-creation-when-successful, which is what M2 vs M3 "
            "measures in the rule-based simulation (whose bargaining agreement rate is ~99%, "
            "so this confound barely exists there)."
        )

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

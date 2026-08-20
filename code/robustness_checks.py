"""
Robustness and sensitivity checks added during revision, in response to
peer-review comments that the paper's headline numbers come from a single,
unvalidated parameter calibration, and that the "M4 nests M2/M3" claim was
asserted but never tested.

Three checks:
  1. M4-nests-M2/M3: run M4 with COMPETITIVE_K=1 (a single solicited seller)
     and confirm it reproduces M2/M3 outcomes exactly, as the paper claims.
  2. Bundle-cost-fraction sensitivity: re-run the M2-vs-M3 welfare-gain
     comparison under BUNDLE_PARTNER_COST_FRAC in {0.35, 0.55, 0.75, 0.95}
     to check whether H3's qualitative direction (and rough magnitude)
     survive reasonable alternative calibrations of how expensive a bundle
     is to provide relative to its value to the buyer.
  3. Buyer-agent-provider penetration sweep: re-run the monopsony
     comparison (common-provider buyers' CS share vs. independent buyers')
     under common_provider_rate in {0.3, 0.5, 0.7, 0.9} to convert the
     paper's speculation that the effect "could plausibly grow" at higher
     penetration into an actual data point.

Uses smaller replicate/buyer counts than the main dataset (this is a
supplementary check, not a replacement for the main 105,600-row dataset)
and never touches simulate_market.py's own default behavior (only passes
explicit non-default arguments here).

Outputs: data/robustness_bundle_frac.csv, data/robustness_monopsony_sweep.csv,
and a printed confirmation of the M4 nesting check.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import simulate_market as sm

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

CHECK_SEED = 20260821
N_REPLICATES_CHECK = 8
N_BUYERS_CHECK = 60


def check_m4_nesting():
    """Confirm that M4 with COMPETITIVE_K=1 reproduces M2 (price-only) and
    M3 (multi-attribute) transaction-by-transaction, as claimed in the
    manuscript (Section 3.2)."""
    rng = np.random.default_rng(CHECK_SEED)
    sellers = sm.make_sellers(rng, concentration="independent_sellers")
    buyers = sm.make_buyers(rng, 30, buyer_provider="independent_buyers")

    orig_k = sm.COMPETITIVE_K
    sm.COMPETITIVE_K = 1
    try:
        mismatches_price_only = 0
        mismatches_multiattr = 0
        for buyer in buyers:
            rng_m2 = np.random.default_rng(hash((buyer.bid, "m2")) % (2**32))
            rng_m3 = np.random.default_rng(hash((buyer.bid, "m3")) % (2**32))
            rng_m4_po = np.random.default_rng(hash((buyer.bid, "m2")) % (2**32))
            rng_m4_ma = np.random.default_rng(hash((buyer.bid, "m3")) % (2**32))

            out_m2 = sm.run_transaction(buyer, sellers, "M2_bilateral_price",
                                         "low", "price_only", 0.0, rng_m2)
            out_m4_po = sm.run_transaction(buyer, sellers, "M4_competitive",
                                            "low", "price_only", 0.0, rng_m4_po)
            out_m3 = sm.run_transaction(buyer, sellers, "M3_bilateral_multiattr",
                                         "low", "multi_attribute", 0.0, rng_m3)
            out_m4_ma = sm.run_transaction(buyer, sellers, "M4_competitive",
                                            "low", "multi_attribute", 0.0, rng_m4_ma)

            if abs(out_m2["price"] - out_m4_po["price"]) > 1e-9:
                mismatches_price_only += 1
            if abs(out_m3["welfare"] - out_m4_ma["welfare"]) > 1e-9:
                mismatches_multiattr += 1
    finally:
        sm.COMPETITIVE_K = orig_k

    n = len(buyers)
    print(f"M4(K=1) vs M2 price mismatches: {mismatches_price_only}/{n}")
    print(f"M4(K=1) vs M3 welfare mismatches: {mismatches_multiattr}/{n}")
    return mismatches_price_only, mismatches_multiattr, n


def _run_reduced_experiment(n_replicates, n_buyers, seed):
    """A reduced version of run_experiment restricted to independent
    sellers/buyers and both bargaining regimes, for sensitivity sweeps
    where we don't need the full factorial design."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_replicates):
        sellers = sm.make_sellers(rng, concentration="independent_sellers")
        buyers = sm.make_buyers(rng, n_buyers, buyer_provider="independent_buyers")
        for buyer in buyers:
            for regime, multiattr in [("M2_bilateral_price", "price_only"),
                                       ("M3_bilateral_multiattr", "price_only"),
                                       ("M3_bilateral_multiattr", "multi_attribute")]:
                out = sm.run_transaction(buyer, sellers, regime, "low", multiattr, 0.0, rng)
                rows.append(dict(replicate=rep, regime=regime, multiattribute=multiattr, **out))
    return pd.DataFrame(rows)


def bundle_cost_frac_sensitivity(fracs=(0.35, 0.55, 0.75, 0.95)):
    results = []
    orig_frac = sm.BUNDLE_PARTNER_COST_FRAC
    try:
        for frac in fracs:
            sm.BUNDLE_PARTNER_COST_FRAC = frac
            df = _run_reduced_experiment(N_REPLICATES_CHECK, N_BUYERS_CHECK, CHECK_SEED)
            m2 = df[df.regime == "M2_bilateral_price"]["welfare"].mean()
            m3_price = df[(df.regime == "M3_bilateral_multiattr") & (df.multiattribute == "price_only")]["welfare"].mean()
            m3_multi = df[(df.regime == "M3_bilateral_multiattr") & (df.multiattribute == "multi_attribute")]["welfare"].mean()
            results.append(dict(bundle_cost_frac=frac, welfare_m2=m2,
                                 welfare_m3_price_only=m3_price,
                                 welfare_m3_multiattr=m3_multi,
                                 welfare_gain_multiattr_vs_price_only=m3_multi - m2))
    finally:
        sm.BUNDLE_PARTNER_COST_FRAC = orig_frac
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(DATA_DIR, "robustness_bundle_frac.csv"), index=False)
    print(out.to_string(index=False))
    return out


def monopsony_penetration_sweep(rates=(0.3, 0.5, 0.7, 0.9)):
    results = []
    for rate in rates:
        rng = np.random.default_rng(CHECK_SEED)
        rows = []
        for rep in range(N_REPLICATES_CHECK):
            sellers = sm.make_sellers(rng, concentration="independent_sellers")
            buyers = sm.make_buyers(rng, N_BUYERS_CHECK, buyer_provider="common_buyer_provider",
                                     common_provider_rate=rate)
            for buyer in buyers:
                monopsony_power = 0.18 if buyer.common_provider else 0.0
                out = sm.run_transaction(buyer, sellers, "M2_bilateral_price", "low",
                                          "price_only", monopsony_power, rng)
                rows.append(dict(replicate=rep, buyer_common_provider=buyer.common_provider, **out))
        df = pd.DataFrame(rows)
        df["cs_share"] = df["consumer_surplus"] / (df["consumer_surplus"] + df["seller_surplus"])
        cs_share_market = df["cs_share"].mean()
        cs_share_flagged = df[df.buyer_common_provider]["cs_share"].mean()
        cs_share_unflagged = df[~df.buyer_common_provider]["cs_share"].mean()
        results.append(dict(common_provider_rate=rate,
                             cs_share_market_avg=cs_share_market,
                             cs_share_flagged_buyers=cs_share_flagged,
                             cs_share_unflagged_buyers=cs_share_unflagged))
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(DATA_DIR, "robustness_monopsony_sweep.csv"), index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    print("=== Check 1: M4(K=1) nests M2/M3 ===")
    check_m4_nesting()
    print("\n=== Check 2: bundle-cost-fraction sensitivity ===")
    bundle_cost_frac_sensitivity()
    print("\n=== Check 3: buyer-agent-provider penetration sweep ===")
    monopsony_penetration_sweep()

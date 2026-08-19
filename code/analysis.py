"""
Analysis of the agentic-commerce market simulation dataset.
Produces every number, table, and statistic reported in the manuscript.
Run after simulate_market.py has written the CSV dataset.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA = os.path.join(_DATA_DIR, "agentic_market_transactions.csv")
OUT = _DATA_DIR + os.sep

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

df = pd.read_csv(DATA)
df["market_id"] = (df.seller_concentration + "|" + df.buyer_concentration + "|" +
                    df.replicate.astype(str))
df["cs_share"] = df.consumer_surplus / df.welfare
df["markup_over_cost"] = df.price / (df.price - df.consumer_surplus - df.bundle_value_realized +
                                      df.bundle_cost_realized) - 1  # not used; recompute directly below

# recompute seller marginal cost implied by identity: price - PS - bundle_cost = cost
df["seller_cost_implied"] = df.price - df.seller_surplus - df.bundle_cost_realized
df["markup"] = (df.price - df.seller_cost_implied) / df.seller_cost_implied

results = {}

# ---------------------------------------------------------------
# Table 1: regime-level summary statistics
# ---------------------------------------------------------------
t1 = df.groupby("regime").agg(
    n=("txn_id", "count"),
    price_mean=("price", "mean"), price_sd=("price", "std"),
    cs_mean=("consumer_surplus", "mean"), cs_sd=("consumer_surplus", "std"),
    ps_mean=("seller_surplus", "mean"), ps_sd=("seller_surplus", "std"),
    welfare_mean=("welfare", "mean"), welfare_sd=("welfare", "std"),
    rounds_mean=("rounds", "mean"),
    raw_agree_rate=("raw_agreement", "mean"),
).reindex(["M0_posted", "M1_search", "M2_bilateral_price",
           "M3_bilateral_multiattr", "M4_competitive"])
print("=== TABLE 1: Regime-level outcomes ===")
print(t1.round(2))
t1.round(2).to_csv(OUT + "table1_regime_summary.csv")

# ---------------------------------------------------------------
# Table 2: multi-attribute value creation (M2 vs M3 vs M4, price_only vs multi_attribute)
# ---------------------------------------------------------------
t2 = df[df.regime.isin(["M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])].groupby(
    ["regime", "multiattribute"]).agg(
    n=("txn_id", "count"),
    price_mean=("price", "mean"),
    cs_mean=("consumer_surplus", "mean"),
    ps_mean=("seller_surplus", "mean"),
    welfare_mean=("welfare", "mean"),
    bundle_realized_rate=("bundle_value_realized", lambda x: (x > 0).mean()),
)
print("\n=== TABLE 2: Price-only vs multi-attribute negotiation ===")
print(t2.round(2))
t2.round(2).to_csv(OUT + "table2_multiattribute.csv")

bundle_sub = df[(df.multiattribute == "multi_attribute") & (df.bundle_value_realized > 0)].copy()
bundle_sub["net_bundle_surplus"] = bundle_sub.bundle_value_realized - bundle_sub.bundle_cost_realized
results["bundle_realization_rate"] = float((df[df.multiattribute == "multi_attribute"].bundle_value_realized > 0).mean())
results["avg_net_bundle_surplus_when_realized"] = float(bundle_sub.net_bundle_surplus.mean())
results["welfare_gain_M3_vs_M2"] = float(
    df[(df.regime == "M3_bilateral_multiattr") & (df.multiattribute == "multi_attribute")].welfare.mean()
    - df[(df.regime == "M2_bilateral_price")].welfare.mean()
)

# ---------------------------------------------------------------
# Table 3: information level effects on surplus split (bargaining regimes only)
# ---------------------------------------------------------------
barg = df[df.regime.isin(["M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])].copy()
info_order = ["low", "preference", "history", "high"]
t3 = barg.groupby("info_level").agg(
    n=("txn_id", "count"),
    price_mean=("price", "mean"),
    cs_mean=("consumer_surplus", "mean"),
    ps_mean=("seller_surplus", "mean"),
    cs_share_mean=("cs_share", "mean"),
).reindex(info_order)
print("\n=== TABLE 3: Seller information level and surplus split ===")
print(t3.round(2))
t3.round(2).to_csv(OUT + "table3_information.csv")

# regression: consumer surplus share ~ information dummies, clustered by market
barg["info_level"] = pd.Categorical(barg.info_level, categories=info_order, ordered=True)
m_info = smf.ols("cs_share ~ C(info_level, Treatment(reference='low')) + C(regime)", data=barg).fit(
    cov_type="cluster", cov_kwds={"groups": barg["market_id"]})
print("\n=== Regression: consumer surplus share on information level (clustered SE by market) ===")
print(m_info.summary().tables[1])
with open(OUT + "reg_info_on_csshare.txt", "w") as f:
    f.write(str(m_info.summary()))

# ---------------------------------------------------------------
# Table 4: buyer-agent concentration (monopsony) effect
# ---------------------------------------------------------------
t4 = barg.groupby("buyer_concentration").agg(
    n=("txn_id", "count"),
    price_mean=("price", "mean"),
    cs_mean=("consumer_surplus", "mean"),
    ps_mean=("seller_surplus", "mean"),
    cs_share_mean=("cs_share", "mean"),
)
print("\n=== TABLE 4: Buyer-agent-provider concentration and surplus split ===")
print(t4.round(2))
t4.round(2).to_csv(OUT + "table4_buyer_concentration.csv")

m_monops = smf.ols("cs_share ~ C(buyer_concentration) + C(regime) + C(info_level)", data=barg).fit(
    cov_type="cluster", cov_kwds={"groups": barg["market_id"]})
with open(OUT + "reg_monopsony_on_csshare.txt", "w") as f:
    f.write(str(m_monops.summary()))
print("\n=== Regression: consumer surplus share on buyer-agent concentration ===")
print(m_monops.summary().tables[1])

# ---------------------------------------------------------------
# Table 5: seller-agent concentration -> markup dispersion (illustrative mechanism)
# ---------------------------------------------------------------
disp = df.groupby(["market_id", "seller_concentration"]).agg(
    markup_sd=("markup", "std"),
    markup_mean=("markup", "mean"),
    price_cv=("price", lambda x: x.std() / x.mean()),
).reset_index()
t5 = disp.groupby("seller_concentration")[["markup_sd", "markup_mean", "price_cv"]].mean()
print("\n=== TABLE 5: Seller-agent-provider concentration and within-market price/markup dispersion ===")
print(t5.round(4))
t5.round(4).to_csv(OUT + "table5_seller_concentration.csv")

# HHI of seller win-shares within each market x regime
def hhi(shares):
    return float((shares ** 2).sum())

win_shares = (df.groupby(["market_id", "seller_concentration", "regime", "seller_id_full"])
              .size().reset_index(name="wins"))
win_shares["total"] = win_shares.groupby(["market_id", "regime"])["wins"].transform("sum")
win_shares["share"] = win_shares["wins"] / win_shares["total"]
hhi_tab = (win_shares.groupby(["market_id", "seller_concentration", "regime"])["share"]
           .apply(hhi).reset_index(name="hhi"))
t_hhi = hhi_tab.groupby(["regime", "seller_concentration"])["hhi"].mean().unstack()
print("\n=== Seller win-share HHI by regime and seller-agent concentration ===")
print(t_hhi.round(4))
t_hhi.round(4).to_csv(OUT + "table_hhi.csv")

# ---------------------------------------------------------------
# Main regression: welfare and price on full factorial design
# ---------------------------------------------------------------
df["regime"] = pd.Categorical(df.regime, categories=[
    "M0_posted", "M1_search", "M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])
m_welfare = smf.ols(
    "welfare ~ C(regime, Treatment(reference='M0_posted')) + C(multiattribute) + "
    "C(seller_concentration) + C(buyer_concentration)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["market_id"]})
print("\n=== Main regression: total welfare ===")
print(m_welfare.summary().tables[1])
with open(OUT + "reg_main_welfare.txt", "w") as f:
    f.write(str(m_welfare.summary()))

m_price = smf.ols(
    "price ~ C(regime, Treatment(reference='M0_posted')) + C(multiattribute) + "
    "C(info_level, Treatment(reference='low')) + C(seller_concentration) + C(buyer_concentration)",
    data=df).fit(cov_type="cluster", cov_kwds={"groups": df["market_id"]})
print("\n=== Main regression: transaction price ===")
print(m_price.summary().tables[1])
with open(OUT + "reg_main_price.txt", "w") as f:
    f.write(str(m_price.summary()))

# ---------------------------------------------------------------
# Save headline numbers used directly in the manuscript prose
# ---------------------------------------------------------------
results["welfare_M0"] = float(df[df.regime == "M0_posted"].welfare.mean())
results["welfare_M1"] = float(df[df.regime == "M1_search"].welfare.mean())
results["welfare_M2"] = float(df[df.regime == "M2_bilateral_price"].welfare.mean())
results["welfare_M3_multiattr"] = float(df[(df.regime == "M3_bilateral_multiattr") & (df.multiattribute == "multi_attribute")].welfare.mean())
results["welfare_M4_multiattr"] = float(df[(df.regime == "M4_competitive") & (df.multiattribute == "multi_attribute")].welfare.mean())
results["price_M0"] = float(df[df.regime == "M0_posted"].price.mean())
results["price_M1"] = float(df[df.regime == "M1_search"].price.mean())
results["price_M2"] = float(df[df.regime == "M2_bilateral_price"].price.mean())
results["cs_share_low_info"] = float(barg[barg.info_level == "low"].cs_share.mean())
results["cs_share_high_info"] = float(barg[barg.info_level == "high"].cs_share.mean())
results["cs_low_info"] = float(barg[barg.info_level == "low"].consumer_surplus.mean())
results["cs_high_info"] = float(barg[barg.info_level == "high"].consumer_surplus.mean())
results["cs_share_independent_buyers"] = float(barg[barg.buyer_concentration == "independent_buyers"].cs_share.mean())
results["cs_share_common_buyer_provider"] = float(barg[barg.buyer_concentration == "common_buyer_provider"].cs_share.mean())
results["markup_sd_independent_sellers"] = float(t5.loc["independent_sellers", "markup_sd"])
results["markup_sd_shared_seller_provider"] = float(t5.loc["shared_seller_provider", "markup_sd"])
results["price_cv_independent_sellers"] = float(t5.loc["independent_sellers", "price_cv"])
results["price_cv_shared_seller_provider"] = float(t5.loc["shared_seller_provider", "price_cv"])
# within-provider-cluster correlation of pricing strategy (ICC)
shared = df[df.seller_concentration == "shared_seller_provider"]
seller_avg = shared.groupby(["market_id", "seller_id_full", "provider_group"])["markup"].mean().reset_index()

def icc_by_market(g):
    overall_var = g["markup"].var(ddof=0)
    if overall_var == 0 or len(g) < 3:
        return np.nan
    grp_means = g.groupby("provider_group")["markup"].transform("mean")
    between_var = ((grp_means - g["markup"].mean()) ** 2).mean()
    return between_var / overall_var

iccs = seller_avg.groupby("market_id").apply(icc_by_market)
results["provider_group_markup_variance_share"] = float(iccs.mean())
results["n_transactions_total"] = int(len(df))
results["n_replicate_markets"] = int(df.market_id.nunique())
results["raw_agreement_rate_bargaining"] = float(barg.raw_agreement.mean())
results["mean_rounds_bargaining"] = float(barg[barg.rounds > 0].rounds.mean())

with open(OUT + "headline_numbers.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== HEADLINE NUMBERS ===")
for k, v in results.items():
    print(f"{k}: {v}")

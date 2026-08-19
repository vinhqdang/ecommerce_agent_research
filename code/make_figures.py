import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
})

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(_ROOT, "data", "agentic_market_transactions.csv")
FIGDIR = os.path.join(_ROOT, "figures") + os.sep

df = pd.read_csv(DATA)
df["seller_cost_implied"] = df.price - df.seller_surplus - df.bundle_cost_realized
df["markup"] = (df.price - df.seller_cost_implied) / df.seller_cost_implied

regime_labels = {
    "M0_posted": "M0\nPosted price\n(limited search)",
    "M1_search": "M1\nAI search\n(no bargaining)",
    "M2_bilateral_price": "M2\nBilateral\n(price only)",
    "M3_bilateral_multiattr": "M3\nBilateral\n(multi-attribute)",
    "M4_competitive": "M4\nCompetitive\n(multi-seller)",
}
regimes = list(regime_labels.keys())

# ---------------------------------------------------------------
# Figure 1: Consumer surplus and seller surplus by regime (stacked)
# ---------------------------------------------------------------
g1 = df.groupby("regime")[["consumer_surplus", "seller_surplus"]].mean().reindex(regimes)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(regimes))
width = 0.55
b1 = ax.bar(x, g1.consumer_surplus, width, label="Consumer surplus", color="#3366a3")
b2 = ax.bar(x, g1.seller_surplus, width, bottom=g1.consumer_surplus,
            label="Seller surplus", color="#e08a2c")
for i, (cs, ps) in enumerate(zip(g1.consumer_surplus, g1.seller_surplus)):
    ax.text(i, cs + ps + 6, f"{cs+ps:,.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([regime_labels[r] for r in regimes], fontsize=8.7)
ax.set_ylabel("Mean surplus per transaction (US\\$)")
ax.set_title("Total welfare and its distribution across market regimes")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, 560)
fig.tight_layout()
fig.savefig(FIGDIR + "fig1_welfare_by_regime.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 2: Consumer surplus share vs seller information level
# ---------------------------------------------------------------
barg = df[df.regime.isin(["M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])].copy()
barg["cs_share"] = barg.consumer_surplus / barg.welfare
info_order = ["low", "preference", "history", "high"]
info_labels = {"low": "Low\n(product/cost only)", "preference": "Preference-\naware",
               "history": "History-\naware", "high": "High\n(urgency + history\n+ alternatives)"}
g2 = barg.groupby("info_level")["cs_share"].agg(["mean", "sem"]).reindex(info_order)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.arange(len(info_order))
ax.bar(x, g2["mean"], yerr=1.96 * g2["sem"], color="#3f7f5f", width=0.55, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([info_labels[i] for i in info_order], fontsize=9)
ax.set_ylabel("Consumer share of total transaction surplus")
ax.set_title("Seller information about the buyer and the consumer surplus share")
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(FIGDIR + "fig2_information_csshare.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 3: Multi-attribute value creation
# ---------------------------------------------------------------
m23 = df[df.regime.isin(["M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])].copy()
g3 = m23.groupby(["regime", "multiattribute"])["welfare"].mean().unstack()
g3 = g3.reindex(["M2_bilateral_price", "M3_bilateral_multiattr", "M4_competitive"])

fig, ax = plt.subplots(figsize=(6.8, 4.2))
x = np.arange(len(g3.index))
width = 0.32
po = g3["price_only"].values
ma = g3["multi_attribute"].values
# M2 has no multi_attribute condition, but NaN is fine (bar will be absent)
ax.bar(x - width/2, po, width, label="Price-only negotiation", color="#9c4141")
ax.bar(x + width/2, ma, width, label="Multi-attribute negotiation\n(price + bundle/gift/delivery)", color="#3366a3")
ax.set_xticks(x)
ax.set_xticklabels(["M2\nBilateral", "M3\nBilateral", "M4\nCompetitive"])
ax.set_ylabel("Mean total welfare per transaction (US\\$)")
ax.set_title("Multi-attribute negotiation and total welfare")
ax.set_ylim(400, 470)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGDIR + "fig3_multiattribute_welfare.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 4: Seller win-share concentration (HHI) by regime and seller-agent concentration
# ---------------------------------------------------------------
df["market_id"] = df.seller_concentration + "|" + df.buyer_concentration + "|" + df.replicate.astype(str)
win_shares = (df.groupby(["market_id", "seller_concentration", "regime", "seller_id_full"])
              .size().reset_index(name="wins"))
win_shares["total"] = win_shares.groupby(["market_id", "regime"])["wins"].transform("sum")
win_shares["share"] = win_shares["wins"] / win_shares["total"]

def hhi(s):
    return float((s ** 2).sum())

hhi_tab = (win_shares.groupby(["market_id", "seller_concentration", "regime"])["share"]
           .apply(hhi).reset_index(name="hhi"))
g4 = hhi_tab.groupby(["regime", "seller_concentration"])["hhi"].mean().unstack().reindex(regimes)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(regimes))
width = 0.35
ax.bar(x - width/2, g4["independent_sellers"], width, label="Independent seller agents", color="#3f7f5f")
ax.bar(x + width/2, g4["shared_seller_provider"], width, label="Shared seller-agent provider", color="#9c4141")
ax.set_xticks(x)
ax.set_xticklabels([regime_labels[r] for r in regimes], fontsize=8.7)
ax.set_ylabel("Mean seller win-share HHI within a market")
ax.set_title("Market concentration across institutional regimes")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(FIGDIR + "fig4_hhi_by_regime.pdf")
plt.close(fig)

print("Figures written to", FIGDIR)
import os
print(os.listdir(FIGDIR))

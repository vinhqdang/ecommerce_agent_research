"""
Agentic Commerce Market Simulation
===================================
A controlled, rule-based agent-based-computational-economics (ACE) simulation
of buyer-agent / seller-agent interaction in electronic commerce, used to
generate the dataset analyzed in the manuscript

  "When AI Shops with AI: Buyer-Seller Agent Negotiation and Market
   Outcomes in Agentic Commerce"

IMPORTANT SCOPE NOTE (see manuscript Section 4 for full discussion):
This simulation uses algorithmic (rule-based, parametrized-strategy) buyer
and seller agents, in the tradition of agent-based computational economics
(Tesfatsion 2006) and negotiation-strategy models (Faratin, Sierra &
Jennings, 1998). It does NOT call large language models. Strategy
parameters (concession speed, information sensitivity, bundling
propensity) are calibrated qualitatively to the mechanisms described in
the recent agentic-commerce literature (e.g. Allouah et al. 2026; Zhu et
al. 2025; Bichler 2026) but the numerical results are properties of this
explicit, reproducible model -- not measurements of any deployed AI
system. All reported figures in the manuscript are computed directly from
this code and the resulting CSV dataset.

Author: generated for Quang-Vinh Dang's ECRA submission project.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import itertools
import json

RNG_GLOBAL_SEED = 20260819

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

N_SELLERS = 16          # sellers in the market
BUNDLE_PARTNER_COST_FRAC = 0.55   # incremental cost of a bundle add-on, as frac of its value
T_ROUNDS = 8             # max negotiation rounds
COMPETITIVE_K = 3        # number of sellers solicited simultaneously in regime M4
IDIOSYNCRATIC_MATCH_SD = 14.0  # $ noise in buyer's ranking of otherwise-identical
                                # sellers, representing unmodelled heterogeneity
                                # (fit, shipping distance, minor brand taste).
                                # Without this term, price-ranked search is a
                                # knife-edge Bertrand mechanism that assigns
                                # 100% of buyers to the single cheapest seller
                                # (HHI = 1) whenever products are undifferentiated;
                                # a small idiosyncratic term keeps that
                                # winner-take-all *tendency* visible while
                                # avoiding a mechanically unrealistic outcome.

REGIMES = ["M0_posted", "M1_search", "M2_bilateral_price",
           "M3_bilateral_multiattr", "M4_competitive"]

INFO_LEVELS = ["low", "preference", "history", "high"]

MULTIATTR_LEVELS = ["price_only", "multi_attribute"]

CONCENTRATION_LEVELS = ["independent_sellers", "shared_seller_provider"]

BUYER_PROVIDER_LEVELS = ["independent_buyers", "common_buyer_provider"]


@dataclass
class Seller:
    sid: int
    cost: float
    min_margin: float          # minimal acceptable markup over cost
    markup_posted: float       # posted-price markup over cost
    bundle_value: float        # gross value of this seller's bundle add-on to a buyer who wants it
    shared_provider: bool      # True if seller uses a common third-party seller-agent
    provider_group: int        # id of the shared provider cluster (0 if independent)

    @property
    def bundle_cost(self):
        return BUNDLE_PARTNER_COST_FRAC * self.bundle_value

    @property
    def reservation_floor(self):
        return self.cost * (1.0 + self.min_margin)

    @property
    def posted_price(self):
        return self.cost * (1.0 + self.markup_posted)


def make_sellers(rng, n=N_SELLERS, concentration="independent_sellers"):
    """Instantiate the seller population. Under 'shared_seller_provider',
    60% of sellers are clustered into 2 provider groups that share
    (nearly) identical strategy parameters -- representing common
    third-party seller-agent infrastructure. This lets us test whether
    infrastructure concentration produces correlated pricing even without
    explicit coordination (cf. Calvano et al. 2020; Bichler 2026)."""
    sellers = []
    shared = concentration == "shared_seller_provider"
    n_shared = int(0.6 * n) if shared else 0
    group_assignment = []
    if shared:
        # two provider clusters
        half = n_shared // 2
        group_assignment = [1] * half + [2] * (n_shared - half) + [0] * (n - n_shared)
        rng.shuffle(group_assignment)
    else:
        group_assignment = [0] * n

    # base draws (idiosyncratic)
    base_cost = rng.uniform(520, 640, size=n)
    base_min_margin = rng.uniform(0.06, 0.20, size=n)
    base_markup_posted = base_min_margin + rng.uniform(0.03, 0.14, size=n)
    base_bundle_value = rng.uniform(60, 160, size=n)

    # if shared provider, overwrite strategy parameters (not cost, which is
    # firm-specific) within each provider cluster to a common (small-noise)
    # value, mimicking a common pricing/negotiation algorithm deployed
    # across otherwise-independent merchants.
    group_strategy = {
        1: dict(min_margin=rng.uniform(0.10, 0.14),
                markup_posted=rng.uniform(0.16, 0.20)),
        2: dict(min_margin=rng.uniform(0.07, 0.09),
                markup_posted=rng.uniform(0.10, 0.13)),
    }

    for i in range(n):
        g = group_assignment[i]
        if g in (1, 2):
            min_margin = group_strategy[g]["min_margin"] + rng.normal(0, 0.004)
            markup_posted = group_strategy[g]["markup_posted"] + rng.normal(0, 0.004)
        else:
            min_margin = base_min_margin[i]
            markup_posted = base_markup_posted[i]
        sellers.append(Seller(
            sid=i,
            cost=base_cost[i],
            min_margin=max(0.02, min_margin),
            markup_posted=max(min_margin + 0.01, markup_posted),
            bundle_value=base_bundle_value[i],
            shared_provider=(g != 0),
            provider_group=g,
        ))
    return sellers


@dataclass
class Buyer:
    bid: int
    valuation: float
    budget: float
    urgency: float          # 0-1, higher = more urgent / less patient
    has_history: bool       # loyal / repeat customer signal
    wants_bundle: bool      # has latent demand for a complementary product
    common_provider: bool   # uses a widely-shared buyer-agent provider
    match_noise: np.ndarray = field(default_factory=lambda: np.zeros(N_SELLERS))
    # fixed idiosyncratic (buyer x seller) taste/fit perturbation, held constant
    # across every institutional regime this buyer is exposed to, so that
    # cross-regime comparisons for a given buyer are not confounded by a
    # fresh random re-draw of which seller "happens" to look best.


def make_buyers(rng, n, buyer_provider="independent_buyers", common_provider_rate=0.70):
    valuation = rng.uniform(780, 1180, size=n)
    budget = valuation * rng.uniform(1.00, 1.12, size=n)
    urgency = rng.beta(2, 5, size=n)
    has_history = rng.random(n) < 0.30
    wants_bundle = rng.random(n) < 0.55
    if buyer_provider == "common_buyer_provider":
        common_flag = rng.random(n) < common_provider_rate
    else:
        common_flag = np.zeros(n, dtype=bool)
    buyers = [
        Buyer(i, valuation[i], budget[i], urgency[i], bool(has_history[i]),
              bool(wants_bundle[i]), bool(common_flag[i]),
              match_noise=rng.normal(0, IDIOSYNCRATIC_MATCH_SD, size=N_SELLERS))
        for i in range(n)
    ]
    return buyers


# ----------------------------------------------------------------------
# Information -> seller inference of buyer willingness-to-pay (WTP)
# ----------------------------------------------------------------------

def seller_reservation_with_information(seller: Seller, buyer: Buyer,
                                         info_level: str,
                                         competitor_signal: Optional[float],
                                         monopsony_power: float) -> float:
    """Return the seller's minimum acceptable price given how much it can
    infer about the buyer, and (in the competitive regime) whether it has
    been told about a rival's offer."""
    floor = seller.reservation_floor
    wtp_estimate = seller.cost  # baseline: no signal beyond cost

    if info_level == "low":
        inferred_extra = 0.0
    elif info_level == "preference":
        inferred_extra = 0.15 * (buyer.valuation - seller.cost) * 0.25
    elif info_level == "history":
        inferred_extra = 0.15 * (buyer.valuation - seller.cost) * 0.40
        if buyer.has_history:
            inferred_extra *= 1.25   # loyalty signal -> seller infers stickiness, extracts a bit more
    elif info_level == "high":
        urgency_premium = buyer.urgency * 0.35 * (buyer.valuation - seller.cost)
        history_premium = (0.40 * (buyer.valuation - seller.cost) *
                            (1.25 if buyer.has_history else 1.0))
        inferred_extra = 0.5 * urgency_premium + 0.5 * history_premium
    else:
        inferred_extra = 0.0

    reservation = floor + inferred_extra

    # Monopsony / aggregated buyer-agent demand pulls the floor DOWN
    # (buyer side bargaining power), independent of seller information.
    reservation = reservation - monopsony_power * (reservation - seller.cost)

    # If the seller is told about a rival's lower offer (competitive
    # disclosure), it must compete the reservation back down.
    if competitor_signal is not None:
        reservation = min(reservation, max(seller.cost * 1.01, competitor_signal * 0.995))

    return max(seller.cost * 1.005, reservation)


# ----------------------------------------------------------------------
# Negotiation protocol: monotonic concession bargaining
# ----------------------------------------------------------------------

def negotiate(seller: Seller, buyer: Buyer, info_level: str,
              multiattribute: bool, n_rivals: int,
              competitor_signal: Optional[float],
              monopsony_power: float, rng) -> dict:
    """Simulate one bilateral negotiation between a buyer agent and a
    seller agent using a parametrized monotonic-concession protocol
    (in the spirit of Faratin-Sierra-Jennings negotiation-decision
    functions and the Rubinstein alternating-offers logic), returning the
    outcome (agreement, price, bundle realized, rounds, surpluses)."""

    reservation_S = seller_reservation_with_information(
        seller, buyer, info_level, competitor_signal, monopsony_power)
    initial_S = seller.posted_price

    reservation_B = min(buyer.valuation, buyer.budget)
    # buyer's opening anchor: below market, informed by a noisy cost estimate
    market_cost_estimate = seller.cost * rng.uniform(0.98, 1.05)
    initial_B = market_cost_estimate * rng.uniform(1.00, 1.06)

    # concession speed: more rivals / more disclosed competition -> seller
    # concedes faster; more buyer urgency -> buyer concedes faster (less
    # patient); shared-provider sellers concede at a group-correlated rate.
    base_seller_speed = 0.55 + 0.12 * min(n_rivals - 1, 3)
    if competitor_signal is not None:
        base_seller_speed += 0.20
    seller_speed = np.clip(base_seller_speed + rng.normal(0, 0.05), 0.25, 0.98)

    buyer_speed = np.clip(0.35 + 0.45 * buyer.urgency + rng.normal(0, 0.05), 0.20, 0.95)

    step_B = (reservation_B - initial_B) / T_ROUNDS * buyer_speed
    step_S = (initial_S - reservation_S) / T_ROUNDS * seller_speed

    bundle_realized = 0.0
    bundle_cost_realized = 0.0

    agreement = False
    price = np.nan
    rounds_used = T_ROUNDS

    offer_B, offer_S = initial_B, initial_S
    for t in range(1, T_ROUNDS + 1):
        offer_B = min(reservation_B, initial_B + step_B * t)
        offer_S = max(reservation_S, initial_S - step_S * t)

        # Multi-attribute channel: instead of a pure price step, seller may
        # substitute part of the concession with a bundle add-on once the
        # gap is "close enough" and the buyer has latent bundle demand.
        if multiattribute and buyer.wants_bundle and (offer_S - offer_B) < 0.35 * (initial_S - initial_B):
            # probability & quality of a good bundle match rises with information
            match_quality = {"low": 0.35, "preference": 0.55,
                              "history": 0.70, "high": 0.85}[info_level]
            if rng.random() < match_quality and bundle_realized == 0.0:
                bundle_realized = seller.bundle_value * rng.uniform(0.7, 1.0)
                bundle_cost_realized = seller.bundle_cost * rng.uniform(0.9, 1.1)
                # bundle substitutes for further price concession: seller
                # "pays" via the bundle instead of the price line, so its
                # reservation effectively rises by the incremental margin
                # it saves; buyer's effective reservation rises by bundle value.
                reservation_B = min(buyer.budget, reservation_B + bundle_realized)
                reservation_S = reservation_S - 0.0  # cost of bundle borne separately

        if offer_B >= offer_S:
            agreement = True
            price = (offer_B + offer_S) / 2.0
            rounds_used = t
            break

    if not agreement:
        price = np.nan

    return dict(agreement=agreement, price=price, rounds=rounds_used,
                bundle_value_realized=bundle_realized,
                bundle_cost_realized=bundle_cost_realized,
                seller_reservation=reservation_S, buyer_reservation=reservation_B)


# ----------------------------------------------------------------------
# One full buyer transaction under a given institutional regime
# ----------------------------------------------------------------------

def search_best_seller(buyer, sellers):
    """Rank sellers by posted price plus a small FIXED idiosyncratic
    buyer-specific perturbation (buyer.match_noise), and return the seller
    the buyer's search agent selects. Using a fixed per-buyer noise vector
    (rather than a fresh draw per call) keeps the comparison across
    institutional regimes for the same buyer uncontaminated by a new random
    tie-break each time."""
    scored = [(s.posted_price + buyer.match_noise[s.sid % len(buyer.match_noise)], s)
              for s in sellers]
    return min(scored, key=lambda x: x[0])[1]


def run_transaction(buyer: Buyer, sellers, regime, info_level,
                     multiattribute_flag, monopsony_power, rng):

    multiattribute = multiattribute_flag == "multi_attribute"

    if regime == "M0_posted":
        # human-like limited search: buyer examines only 2 random sellers
        examined = rng.choice(sellers, size=2, replace=False)
        best = min(examined, key=lambda s: s.posted_price)
        price = best.posted_price
        cs = buyer.valuation - price
        ps = price - best.cost
        return dict(agreement=True, raw_agreement=True, price=price, rounds=0, search_breadth=2,
                    seller_id=best.sid, bundle_value_realized=0.0,
                    bundle_cost_realized=0.0, consumer_surplus=cs,
                    seller_surplus=ps, welfare=cs + ps,
                    provider_group=best.provider_group)

    if regime == "M1_search":
        # AI-assisted costless search over the FULL market, still no bargaining
        best = search_best_seller(buyer, sellers)
        price = best.posted_price
        cs = buyer.valuation - price
        ps = price - best.cost
        return dict(agreement=True, raw_agreement=True, price=price, rounds=0, search_breadth=len(sellers),
                    seller_id=best.sid, bundle_value_realized=0.0,
                    bundle_cost_realized=0.0, consumer_surplus=cs,
                    seller_surplus=ps, welfare=cs + ps,
                    provider_group=best.provider_group)

    if regime == "M2_bilateral_price":
        best = search_best_seller(buyer, sellers)
        outcome = negotiate(best, buyer, info_level, multiattribute=False,
                             n_rivals=1, competitor_signal=None,
                             monopsony_power=monopsony_power, rng=rng)
        return _finalize(outcome, best, buyer, search_breadth=len(sellers))

    if regime == "M3_bilateral_multiattr":
        best = search_best_seller(buyer, sellers)
        outcome = negotiate(best, buyer, info_level, multiattribute=multiattribute,
                             n_rivals=1, competitor_signal=None,
                             monopsony_power=monopsony_power,
                             rng=rng)
        return _finalize(outcome, best, buyer, search_breadth=len(sellers))

    if regime == "M4_competitive":
        # buyer agent's search identifies the best posted-price seller (as
        # in M1/M2), then solicits simultaneous offers from that seller
        # plus (COMPETITIVE_K - 1) additional randomly-encountered rivals,
        # and negotiates with all of them in parallel. This nests the
        # single-seller bilateral case (M2/M3) inside a strictly larger
        # competitive candidate set, so any advantage of M4 over M2/M3 is
        # attributable to competitive disclosure/leverage, not to a luckier
        # draw of counterparties.
        cheapest = search_best_seller(buyer, sellers)
        others = [s for s in sellers if s.sid != cheapest.sid]
        extra = list(rng.choice(others, size=COMPETITIVE_K - 1, replace=False))
        rivals = [cheapest] + extra
        outcomes = []
        # sequential solicitation; buyer discloses the best rival price
        # obtained so far to the next seller (strategic disclosure)
        best_rival_price = None
        for s in rivals:
            out = negotiate(s, buyer, info_level, multiattribute=multiattribute,
                             n_rivals=COMPETITIVE_K, competitor_signal=best_rival_price,
                             monopsony_power=monopsony_power, rng=rng)
            outcomes.append((s, out))
            if out["agreement"]:
                if best_rival_price is None or out["price"] < best_rival_price:
                    best_rival_price = out["price"]
        # buyer selects the agreement with highest realized utility
        feasible = [(s, o) for s, o in outcomes if o["agreement"]]
        if not feasible:
            # fallback to best posted price among rivals
            best = search_best_seller(buyer, rivals)
            price = best.posted_price
            cs = buyer.valuation - price
            ps = price - best.cost
            return dict(agreement=True, raw_agreement=False, price=price, rounds=T_ROUNDS, search_breadth=COMPETITIVE_K,
                        seller_id=best.sid, bundle_value_realized=0.0,
                        bundle_cost_realized=0.0, consumer_surplus=cs,
                        seller_surplus=ps, welfare=cs + ps,
                        provider_group=best.provider_group)
        def buyer_utility(item):
            s, o = item
            return (buyer.valuation + o["bundle_value_realized"] - o["price"])
        best_s, best_o = max(feasible, key=buyer_utility)
        return _finalize(best_o, best_s, buyer, search_breadth=COMPETITIVE_K)

    raise ValueError(regime)


def _finalize(outcome, seller, buyer, search_breadth):
    if not outcome["agreement"]:
        # impasse -> fallback to this seller's posted price (buyer still
        # transacts, at the less favourable posted rate)
        price = seller.posted_price
        cs = buyer.valuation - price
        ps = price - seller.cost
        return dict(agreement=True, raw_agreement=False, price=price, rounds=T_ROUNDS,
                    search_breadth=search_breadth,
                    seller_id=seller.sid, bundle_value_realized=0.0,
                    bundle_cost_realized=0.0, consumer_surplus=cs,
                    seller_surplus=ps, welfare=cs + ps,
                    provider_group=seller.provider_group)
    price = outcome["price"]
    bv = outcome["bundle_value_realized"]
    bc = outcome["bundle_cost_realized"]
    cs = buyer.valuation + bv - price
    ps = price - seller.cost - bc
    return dict(agreement=True, raw_agreement=True, price=price, rounds=outcome["rounds"],
                search_breadth=search_breadth, seller_id=seller.sid,
                bundle_value_realized=bv, bundle_cost_realized=bc,
                consumer_surplus=cs, seller_surplus=ps, welfare=cs + ps,
                provider_group=seller.provider_group)


# ----------------------------------------------------------------------
# Full factorial experiment
# ----------------------------------------------------------------------

def conditions_for_regime(regime):
    """Return the list of (info_level, multiattribute) combinations that
    are behaviourally meaningful for a given institutional regime. M0/M1
    involve no negotiation, so information and multi-attribute conditions
    are not applicable and are fixed at placeholder values. M2 is
    price-only by construction. M3 and M4 are crossed with both factors."""
    if regime in ("M0_posted", "M1_search"):
        return [("low", "price_only")]
    if regime == "M2_bilateral_price":
        return [(info, "price_only") for info in INFO_LEVELS]
    if regime in ("M3_bilateral_multiattr", "M4_competitive"):
        return [(info, ma) for info in INFO_LEVELS for ma in MULTIATTR_LEVELS]
    raise ValueError(regime)


def run_experiment(n_replicates=20, n_buyers=60, seed=RNG_GLOBAL_SEED):
    """Matched (within-market) experimental design: for every replicate
    market -- a fixed draw of sellers and buyers under a given
    (seller_concentration, buyer_concentration) macro-condition -- every
    buyer is passed through *every* institutional regime / information /
    multi-attribute condition applicable to that regime. Holding the
    seller and buyer population fixed across treatments removes the
    sampling confound of comparing regimes drawn from different market
    instances, so that price/surplus differences across regimes reflect
    the institutional treatment itself."""
    rng = np.random.default_rng(seed)
    rows = []
    txn_id = 0

    for concentration in CONCENTRATION_LEVELS:
        for buyer_provider in BUYER_PROVIDER_LEVELS:
            for rep in range(n_replicates):
                sellers = make_sellers(rng, concentration=concentration)
                buyers = make_buyers(rng, n_buyers, buyer_provider=buyer_provider)

                for buyer in buyers:
                    # Monopsony/aggregated bargaining power only accrues to
                    # buyers actually on the common provider (buyer.common_provider,
                    # drawn for ~70% of buyers under the "common_buyer_provider"
                    # macro-condition in make_buyers) -- not to every buyer in
                    # that market, so the concentration effect is a per-buyer
                    # mechanism rather than a market-wide toggle.
                    monopsony_power = 0.18 if (buyer_provider == "common_buyer_provider"
                                                and buyer.common_provider) else 0.0
                    for regime in REGIMES:
                        for (info_level, multiattr) in conditions_for_regime(regime):
                            out = run_transaction(buyer, sellers, regime, info_level,
                                                   multiattr, monopsony_power, rng)
                            row = dict(
                                txn_id=txn_id, replicate=rep,
                                regime=regime, info_level=info_level,
                                multiattribute=multiattr,
                                seller_concentration=concentration,
                                buyer_concentration=buyer_provider,
                                buyer_id=f"{concentration}_{buyer_provider}_{rep}_{buyer.bid}",
                                buyer_valuation=buyer.valuation,
                                buyer_urgency=buyer.urgency,
                                buyer_has_history=buyer.has_history,
                                buyer_wants_bundle=buyer.wants_bundle,
                                buyer_common_provider=buyer.common_provider,
                            )
                            row.update(out)
                            row["seller_id_full"] = f"{concentration}_{buyer_provider}_{rep}_{out['seller_id']}"
                            rows.append(row)
                            txn_id += 1

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    df = run_experiment(n_replicates=20, n_buyers=60)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "agentic_market_transactions.csv")
    df.to_csv(out_path, index=False)
    print("rows:", len(df))
    print(df.groupby("regime")[["price", "consumer_surplus", "seller_surplus", "welfare"]].mean())

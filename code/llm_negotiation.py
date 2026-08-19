"""
LLM-Agent Negotiation: a validation layer against the rule-based ACE
simulation in simulate_market.py.

The rule-based simulation is deliberately NOT an LLM: its regimes M2
(bilateral, price-only) and M3 (bilateral, multi-attribute) are
parametrized concession strategies calibrated to the agentic-commerce
literature, not measurements of a deployed AI system. This module closes
that gap for a bounded pilot: real buyer-agent and seller-agent LLMs
(Gemini and free OpenRouter models) negotiate directly, over the SAME
valuation/cost/bundle-value distributions used in simulate_market.py, so
the resulting price/welfare split can be compared to the M2/M3 numbers
in headline_numbers.json.

Design:
  - REGIMES: "price_only" (M2 analog), "multiattr" (M3 analog, seller may
    concede a bundle instead of / alongside price).
  - MODEL PAIRINGS: gemini-vs-gemini, openrouter-vs-openrouter, and a
    mixed gemini(buyer)-vs-openrouter(seller) pairing -- the latter
    speaks directly to the "AI shops with AI" framing: does the outcome
    depend on which LLM plays which role?
  - N_REPLICATES buyer/seller draws are shared across every regime x
    pairing cell (paired design, same spirit as simulate_market.py).
  - This is intentionally small-N (real API calls, free-tier rate
    limits): treat results as an exploratory/pilot robustness check, not
    a powered experiment. Every dropped/failed episode is logged, never
    silently discarded.

Outputs:
  data/llm_negotiation_results.csv       -- one row per episode
  data/llm_negotiation_transcripts.jsonl -- full dialogue per episode
"""

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llm_clients import call_and_parse_json, LLMCallError

RNG_SEED = 20260819
N_REPLICATES = 6
MAX_ROUNDS = 6  # each round = one buyer turn + one seller turn

REGIMES = ["price_only", "multiattr"]
PAIRINGS = [
    ("gemini", "gemini"),
    ("openrouter", "openrouter"),
    ("gemini", "openrouter"),  # buyer=gemini, seller=openrouter
]

BUNDLE_PARTNER_COST_FRAC = 0.55

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
RESULTS_CSV = os.path.join(DATA_DIR, "llm_negotiation_results.csv")
TRANSCRIPTS_JSONL = os.path.join(DATA_DIR, "llm_negotiation_transcripts.jsonl")


def draw_replicates(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    reps = []
    for i in range(n):
        cost = float(rng.uniform(520, 640))
        min_margin = float(rng.uniform(0.06, 0.20))
        bundle_value = float(rng.uniform(60, 160))
        valuation = float(rng.uniform(780, 1180))
        budget = valuation * float(rng.uniform(1.00, 1.12))
        reps.append(dict(
            replicate=i,
            seller_cost=cost,
            seller_floor=cost * (1.0 + min_margin),
            bundle_value=bundle_value,
            bundle_cost=BUNDLE_PARTNER_COST_FRAC * bundle_value,
            buyer_valuation=valuation,
            buyer_budget=budget,
        ))
    return reps


SCHEMA_NOTE = (
    'Respond with ONLY a single JSON object, no markdown fences, no other text:\n'
    '{"action": "propose" | "accept" | "walk", '
    '"price": <number, your proposed price, or null if action is accept/walk>, '
    '"include_bundle": <true|false>, '
    '"message": "<one short sentence to the other party>"}'
)


def buyer_system_prompt(rep, regime):
    bundle_clause = ""
    if regime == "multiattr":
        bundle_clause = (
            f"\nThe seller MAY offer a bundle add-on worth ${rep['bundle_value']:.0f} to you "
            "(e.g. an accessory, extended warranty, or faster delivery). If they include it "
            "in a proposal, set include_bundle accordingly when you reply."
        )
    else:
        bundle_clause = "\nThere is no bundle/add-on in this negotiation -- only price. Always set include_bundle to false."
    return (
        "You are an AI shopping agent negotiating on behalf of a human buyer to purchase one unit "
        "of a product from a seller's AI agent. This is a real economic negotiation: your private "
        "reservation values below are secret and must never be revealed verbatim to the seller.\n"
        f"Your private valuation for the product is ${rep['buyer_valuation']:.2f} (the most you would "
        f"ever be willing to pay). Your budget ceiling is ${rep['buyer_budget']:.2f}. "
        "You want the lowest price possible while still closing a deal if it is worthwhile.\n"
        f"{bundle_clause}\n"
        f"You have at most {MAX_ROUNDS} rounds. If no deal is reached by then, negotiation ends with "
        "no trade (both sides get nothing), so do not stall pointlessly once terms are reasonable.\n"
        f"{SCHEMA_NOTE}"
    )


def seller_system_prompt(rep, regime):
    bundle_clause = ""
    if regime == "multiattr":
        bundle_clause = (
            f"\nYou MAY offer a bundle add-on to the buyer worth ${rep['bundle_value']:.0f} to them, "
            f"which costs you ${rep['bundle_cost']:.2f} to provide. Offering it instead of a pure price "
            "cut can be more profitable for you when it is worth more to the buyer than it costs you -- "
            "use it strategically. Set include_bundle to true only on rounds where you are actually "
            "offering it."
        )
    else:
        bundle_clause = "\nThere is no bundle/add-on in this negotiation -- only price. Always set include_bundle to false."
    return (
        "You are an AI shopping agent negotiating on behalf of a seller (a merchant) to sell one unit "
        "of a product to a buyer's AI agent. This is a real economic negotiation: your private cost "
        "below is secret and must never be revealed verbatim to the buyer.\n"
        f"Your unit cost is ${rep['seller_cost']:.2f}. You should not accept a deal (accounting for any "
        f"bundle cost) that nets you less than about ${rep['seller_floor']:.2f} in price terms, but you "
        "are free to use judgement.\n"
        f"{bundle_clause}\n"
        f"You have at most {MAX_ROUNDS} rounds. If no deal is reached by then, negotiation ends with "
        "no trade (both sides get nothing), so do not stall pointlessly once terms are reasonable.\n"
        f"{SCHEMA_NOTE}"
    )


def format_transcript(history):
    if not history:
        return "(No offers have been made yet. You may open the negotiation.)"
    lines = []
    for turn in history:
        lines.append(
            f"[Round {turn['round']}] {turn['speaker'].upper()} -> action={turn['action']}, "
            f"price={turn['price']}, include_bundle={turn['include_bundle']}, "
            f'says: "{turn["message"]}"'
        )
    return "\n".join(lines)


def run_episode(rep, regime, buyer_provider, seller_provider, log):
    history = []
    last_proposal = None  # (speaker, price, include_bundle)
    deal = False
    deal_price = None
    deal_bundle = False
    schema_violations = 0
    failed = False

    for rnd in range(1, MAX_ROUNDS + 1):
        for speaker, provider, sys_prompt_fn in (
            ("buyer", buyer_provider, buyer_system_prompt),
            ("seller", seller_provider, seller_system_prompt),
        ):
            sys_prompt = sys_prompt_fn(rep, regime)
            user_prompt = (
                f"Negotiation so far:\n{format_transcript(history)}\n\n"
                f"It is round {rnd} of at most {MAX_ROUNDS}. Give your move now."
            )
            try:
                parsed, raw_text = call_and_parse_json(provider, sys_prompt, user_prompt)
            except (LLMCallError, Exception) as e:
                log(f"  EPISODE FAILED at round {rnd} ({speaker}, {provider}): {e}")
                failed = True
                break

            action = str(parsed.get("action", "")).lower().strip()
            price = parsed.get("price", None)
            include_bundle = bool(parsed.get("include_bundle", False))
            message = str(parsed.get("message", ""))[:300]

            if regime == "price_only" and include_bundle:
                schema_violations += 1
                include_bundle = False

            turn = dict(round=rnd, speaker=speaker, action=action, price=price,
                        include_bundle=include_bundle, message=message, raw=raw_text[:500])
            history.append(turn)

            if action == "accept":
                if last_proposal is None:
                    # nothing to accept yet -- treat as a no-op propose-less turn
                    schema_violations += 1
                    continue
                deal = True
                deal_price = last_proposal[1]
                deal_bundle = last_proposal[2]
                break
            elif action == "walk":
                deal = False
                break
            elif action == "propose":
                if price is not None:
                    last_proposal = (speaker, float(price), include_bundle)
                else:
                    schema_violations += 1
            else:
                schema_violations += 1

        if failed or deal or (history and history[-1]["action"] == "walk"):
            break

    outcome = dict(
        replicate=rep["replicate"], regime=regime,
        buyer_provider=buyer_provider, seller_provider=seller_provider,
        deal=deal, failed=failed, rounds_used=history[-1]["round"] if history else 0,
        schema_violations=schema_violations,
        price=deal_price, bundle_included=deal_bundle,
        seller_cost=rep["seller_cost"], seller_floor=rep["seller_floor"],
        buyer_valuation=rep["buyer_valuation"], buyer_budget=rep["buyer_budget"],
        bundle_value=rep["bundle_value"], bundle_cost=rep["bundle_cost"],
    )
    if deal:
        bv = rep["bundle_value"] if deal_bundle else 0.0
        bc = rep["bundle_cost"] if deal_bundle else 0.0
        outcome["consumer_surplus"] = rep["buyer_valuation"] + bv - deal_price
        outcome["producer_surplus"] = deal_price - rep["seller_cost"] - bc
        outcome["welfare"] = outcome["consumer_surplus"] + outcome["producer_surplus"]
    else:
        outcome["consumer_surplus"] = 0.0
        outcome["producer_surplus"] = 0.0
        outcome["welfare"] = 0.0

    return outcome, history


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    replicates = draw_replicates(N_REPLICATES)
    results = []

    def log(msg):
        print(msg, flush=True)

    total = len(replicates) * len(REGIMES) * len(PAIRINGS)
    done = 0
    with open(TRANSCRIPTS_JSONL, "w") as tf:
        for rep in replicates:
            for regime in REGIMES:
                for buyer_provider, seller_provider in PAIRINGS:
                    done += 1
                    tag = f"[{done}/{total}] rep={rep['replicate']} regime={regime} buyer={buyer_provider} seller={seller_provider}"
                    log(f"RUNNING {tag}")
                    t0 = time.time()
                    outcome, history = run_episode(rep, regime, buyer_provider, seller_provider, log)
                    dt = time.time() - t0
                    log(f"  -> deal={outcome['deal']} price={outcome['price']} "
                        f"welfare={outcome['welfare']:.1f} rounds={outcome['rounds_used']} "
                        f"violations={outcome['schema_violations']} ({dt:.1f}s)")
                    results.append(outcome)
                    tf.write(json.dumps(dict(meta=outcome, transcript=history)) + "\n")
                    tf.flush()

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    log(f"\nWrote {len(df)} episodes to {RESULTS_CSV}")
    log(f"Wrote transcripts to {TRANSCRIPTS_JSONL}")
    log(f"Deal rate: {df['deal'].mean():.2%}, failed: {df['failed'].sum()}")


if __name__ == "__main__":
    main()

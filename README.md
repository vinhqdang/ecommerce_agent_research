# When AI Shops with AI

Research code and data for a study of buyer-seller AI agent negotiation and
market outcomes in agentic commerce, prepared for submission to *Electronic
Commerce Research and Applications* (Elsevier).

> **When AI Shops with AI: Buyer-Seller Agent Negotiation and Market
> Outcomes in Agentic Commerce**

## Overview

As AI shopping agents start searching, comparing, and negotiating on
behalf of both buyers and sellers, the institutional rules of an
e-commerce market — how much searching happens, whether prices are fixed
or bargained, whether negotiation can range over price alone or over
bundles and terms, and how concentrated the agents themselves are on
shared infrastructure — start to matter for who captures the resulting
surplus. This project studies that question in two complementary ways:

1. A **rule-based agent-based-computational-economics (ACE) simulation**
   (`code/simulate_market.py`) of buyer and seller agents negotiating
   under five institutional regimes, crossed with seller information
   level, multi-attribute vs. price-only bargaining, and agent-provider
   concentration on both sides of the market. This is the main empirical
   engine behind every headline result below.
2. A smaller **LLM-agent negotiation pilot** (`code/llm_negotiation.py`)
   in which real Gemini and OpenRouter models play the buyer and seller
   roles directly, used as an exploratory robustness check on whether the
   simulation's mechanism-level predictions hold up with actual language
   models doing the negotiating.

## Repository structure

```
code/
  simulate_market.py            # rule-based ACE market simulation (main engine)
  analysis.py                   # statistics, regressions, headline numbers
  make_figures.py                # the 4 publication figures
  llm_clients.py                 # Gemini + OpenRouter API wrappers (retry/key rotation)
  llm_negotiation.py              # real LLM buyer/seller negotiation pilot
  analyze_llm_negotiation.py       # LLM pilot vs. rule-based comparison
data/
  agentic_market_transactions.csv    # 105,600-row simulated transaction dataset
  headline_numbers.json              # every number cited in the manuscript
  table1..6_*.csv, table_hhi.csv     # summary tables
  reg_*.txt                          # regression output (cluster-robust SEs)
  llm_negotiation_results.csv         # one row per LLM-pilot episode
  llm_negotiation_transcripts.jsonl    # full LLM-pilot negotiation dialogues
  llm_validation_summary.json          # LLM pilot vs. rule-based comparison
figures/
  fig1..4_*.pdf, *-1.png          # vector figures + PNG previews
tex/
  references.bib                 # bibliography
  manuscript.tex                 # the manuscript (in progress)
```

## Methodology

### Rule-based market simulation

Buyer and seller agents negotiate under five institutional regimes:

| Regime | Description |
|---|---|
| M0 | Posted price, limited human-like search (2 of 16 sellers examined) |
| M1 | AI-assisted full-market search, no bargaining |
| M2 | Bilateral bargaining, price-only |
| M3 | Bilateral bargaining, multi-attribute (bundles/gifts/delivery allowed) |
| M4 | Competitive: simultaneous multi-seller solicitation with rival disclosure |

These are crossed with seller information about the buyer (low /
preference-aware / history-aware / high), multi-attribute vs. price-only
negotiation, seller-agent-provider concentration (independent sellers vs.
sellers sharing third-party pricing infrastructure), and buyer-agent
concentration (independent buyers vs. buyers pooled on a common
buyer-agent provider, which gives them aggregated bargaining/monopsony
power).

The design is a **matched/paired experiment**: for every replicate market
(a fixed draw of sellers and buyers), the same buyers are run through
every applicable regime and condition, so cross-regime comparisons are
not confounded by differing underlying populations. The negotiation
protocol follows a parametrized monotonic-concession model in the
tradition of Faratin, Sierra & Jennings (1998) and Rubinstein-style
alternating offers; the overall design follows the agent-based
computational economics tradition (Tesfatsion, 2006). It does **not**
call any LLM — strategy parameters are calibrated qualitatively to the
mechanisms described in the recent agentic-commerce literature, but the
numerical results are properties of this explicit, reproducible model,
not measurements of a deployed AI system.

Dataset: `data/agentic_market_transactions.csv`, 105,600 rows across 80
replicate markets, deterministically reproducible via
`python3 code/simulate_market.py` (fixed seed).

### LLM-agent negotiation pilot

A separate, much smaller pilot has real Gemini (`gemini-flash-lite-latest`)
and free OpenRouter models negotiate directly, playing the buyer and
seller roles over a JSON action protocol (`propose` / `accept` / `walk`),
across the same valuation/cost/bundle-value distributions used in the
main simulation. Two regimes (price-only and multi-attribute, analogous
to M2/M3) are crossed with three model pairings (Gemini vs. Gemini,
OpenRouter vs. OpenRouter, and a mixed Gemini-buyer/OpenRouter-seller
pairing) across 6 paired replicate draws, for 36 episodes total.

This is treated throughout as an **exploratory, small-N robustness
check**, not a substitute for the rule-based engine: real free-tier API
calls occasionally fail to return parseable output (6 of 36 episodes
here; every failure is logged and never silently dropped), so the
deal-conditional comparison is reported alongside the unconditional one
to avoid conflating negotiation breakdown with surplus creation. See
`data/llm_validation_summary.json`.

## Key results

See `data/headline_numbers.json` for exact figures; all numbers below are
from the current, reproducible dataset.

- **Search vs. bargaining vs. bundling.** Welfare rises from 416 (M0,
  limited search) to 443 (M1/M2 — search improves matching efficiency;
  pure bilateral bargaining is purely redistributive, since M1 and M2
  welfare are numerically identical) to 456 (M3, multi-attribute
  bargaining genuinely creates additional surplus: +13.6 welfare units,
  with the bundle realized about 47% of the time and an average net
  bundle surplus of $35.8 when it is).
- **Personalization and extraction.** As seller information about the
  buyer rises from low to high, consumer surplus share falls from 84% to
  68% (regression coefficient on "high info" = −0.160, p < 0.001,
  clustered standard errors).
- **Search-driven concentration.** The HHI of seller win-shares rises
  from about 0.10 under limited human-like search to about 0.60 under
  full AI-mediated search, but competitive multi-seller solicitation
  (M4) moderates this back down to about 0.45 — a non-monotonic effect
  of adding competitive pressure back into a highly efficient search
  regime.
- **Buyer-side concentration (monopsony).** Buyers on a common
  buyer-agent provider capture a higher consumer surplus share than
  independent buyers (0.80 vs. 0.79, significant at p = 0.004 after
  controls). This is a genuinely per-buyer mechanism — only the ~70% of
  buyers actually flagged as being on the shared provider receive the
  monopsony discount in a given market.
- **Seller-agent infrastructure concentration.** Does not mechanically
  reduce aggregate market-wide price dispersion (a market with a few
  large but distinct providers can be as heterogeneous as one with many
  independent sellers), but shared-provider membership does explain
  about 45% of the cross-seller variance in average markup within those
  markets — real within-network price correlation exists even though it
  doesn't show up as reduced market-wide dispersion. Worth stating
  carefully rather than overclaiming "AI causes collusion."
- **LLM-pilot corroboration.** The deal-conditional welfare gain from
  multi-attribute vs. price-only negotiation in the LLM pilot (+30.4
  units) agrees directionally with the rule-based simulation's M3-vs-M2
  finding (+13.4 units), a small but real piece of corroborating evidence
  that the mechanism isn't an artifact of the rule-based model's
  particular functional form.

## Reproducing the results

```bash
# regenerate the simulation dataset (deterministic, ~10s)
python3 code/simulate_market.py

# regenerate all tables/regressions/headline numbers
python3 code/analysis.py

# regenerate the 4 publication figures
python3 code/make_figures.py

# regenerate the LLM-agent negotiation pilot (real API calls, ~15-20 min)
python3 code/llm_negotiation.py
python3 code/analyze_llm_negotiation.py
```

Requirements: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels` for
the simulation/analysis/figures; additionally `requests`, `python-dotenv`,
`tenacity` for the LLM pilot, plus a local, git-ignored `.env` file with
`GEMINI_API_KEY` and `OPENROUTER_API_KEY` (`OPENROUTER_API_KEY_2`, etc.
for key rotation across multiple free-tier quotas).

For the LaTeX build: a TeX distribution with `elsarticle.cls` (TeX Live
`texlive-publishers` package, or the MiKTeX equivalent) plus `natbib`.

## Manuscript status

The empirical engine (simulation + analysis + figures + LLM pilot) is
complete and reproducible. The manuscript itself is in progress in
`tex/manuscript.tex`, targeting the ECRA author guidelines: title page,
abstract, keywords, highlights, introduction, literature review, formal
model and hypotheses, methodology, results, discussion, limitations,
Declaration of Generative AI use, CRediT statement, competing interests,
funding, and data availability statements.

## Data and code availability

The dataset (`data/agentic_market_transactions.csv`) and all analysis/
generation code in this repository together constitute the full data
and code needed to reproduce every table, regression, and figure in the
manuscript.

## Author

Quang-Vinh Dang

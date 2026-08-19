# Agentic Commerce ECRA Manuscript — Handoff Package

Status as of 2026-08-19. This package contains everything produced so far
for the manuscript:

  "When AI Shops with AI: Buyer-Seller Agent Negotiation and Market
   Outcomes in Agentic Commerce"

targeted at *Electronic Commerce Research and Applications* (Elsevier).

## What's DONE

### 1. A real, working computational market simulation (`code/simulate_market.py`)
Rule-based agent-based-computational-economics (ACE) model — NOT calls to
any LLM (no API access was available in the sandbox this was built in).
Buyer agents and seller agents negotiate under five institutional
regimes:

- **M0** posted price, limited human-like search (examine 2 of 16 sellers)
- **M1** AI-assisted full-market search, no bargaining
- **M2** bilateral bargaining, price-only, with a single seller (the one
  full search identified as cheapest)
- **M3** bilateral bargaining, multi-attribute allowed (seller can offer
  bundles/gifts/delivery instead of pure price concessions)
- **M4** competitive: buyer solicits offers from multiple sellers
  simultaneously and can disclose rival offers

Crossed with: seller information level about the buyer (low /
preference-aware / history-aware / high), multi-attribute vs price-only,
seller-agent-provider concentration (independent sellers vs. sellers
clustered on shared third-party pricing infrastructure), and buyer-agent
concentration (independent buyers vs. many buyers on one common
buyer-agent provider, giving aggregated bargaining/monopsony power).

**Design is a matched/paired experiment**: for every replicate market
(fixed draw of sellers + buyers), the SAME buyers are run through every
applicable regime/condition, so cross-regime comparisons are not
confounded by different underlying populations. This was fixed after an
initial bug (sellers were being redrawn per cell) — see the git-style
history in the code comments / commit messages if you keep this under
version control locally.

Also fixed: an initial version made price-ranked full-market search a
deterministic Bertrand mechanism that assigned 100% of buyers in a market
to the single cheapest seller (HHI=1.0, unrealistic knife-edge). Added a
small fixed per-buyer idiosyncratic taste/fit term (`IDIOSYNCRATIC_MATCH_SD`)
so this becomes a strong-but-not-total winner-take-all tendency instead.

Output: `data/agentic_market_transactions.csv` — **105,600 rows**, 80
replicate markets. Fully reproducible: `python3 code/simulate_market.py`
regenerates it deterministically (fixed seed `RNG_GLOBAL_SEED`).

### 2. Full statistical analysis (`code/analysis.py`)
Produces every table and regression referenced below. Outputs are in
`data/`: `table1..5_*.csv`, `table_hhi.csv`, `reg_*.txt` (statsmodels
OLS output, cluster-robust SEs by replicate market), and
`headline_numbers.json` (every number you'd want to cite in prose).

### 3. Four publication figures (`code/make_figures.py`, `figures/*.pdf` + PNG previews)
1. `fig1_welfare_by_regime` — stacked CS/PS by regime
2. `fig2_information_csshare` — consumer surplus share vs. seller
   information level
3. `fig3_multiattribute_welfare` — price-only vs multi-attribute welfare
   by regime
4. `fig4_hhi_by_regime` — seller win-share HHI by regime × concentration

### 4. Verified bibliography (`tex/references.bib`)
~28 references with real, checked DOIs/identifiers spanning: classic
bargaining/transaction-cost/search theory (Coase, Rubinstein, Nash,
Bakos, Stigler, Akerlof), price discrimination (Bergemann-Brooks-Morris,
Dubé-Misra, Acquisti-Varian), algorithmic pricing/collusion (Calvano et
al. 2020 AER + follow-ups, Klein 2021, Brown & MacKay 2023), and the very
latest 2025–2026 agentic-commerce literature (Bichler 2026 *Electronic
Markets*, Shahidi et al.'s NBER "Coasean Singularity" chapter, Allouah et
al.'s ACES/WWW'26 paper, Bansal et al.'s Magentic Marketplace, Zhu et al.
2025, Bostoen & Krämer 2026). A few arXiv preprints are flagged `TODO`
— check whether they have since appeared in a formal venue before final
submission.

## Key results already generated (see `data/headline_numbers.json` for exact figures)

Updated 2026-08-19 after a code-review-flagged fix (see "Data integrity fix"
below): a buyer-level monopsony flag was wired into the negotiation logic
correctly (it was previously drawn but silently ignored). All numbers below
reflect the corrected, regenerated dataset; all qualitative findings are
unchanged, magnitudes shifted only slightly except where noted.

- **Search vs. bargaining vs. bundling**: welfare(M0)=416 < welfare(M1)=
  welfare(M2)=443 < welfare(M3, multi-attr)=456 ≈ welfare(M4, multi-attr)=
  455. Clean theoretical story: search improves *matching* efficiency;
  pure bilateral bargaining is redistributive (M1 and M2 welfare are
  numerically identical — bargaining only reallocates the same pie);
  multi-attribute negotiation genuinely *creates* additional surplus
  (+13.6 welfare units, bundle realized ~47% of the time, average net
  bundle surplus ≈ $35.8 when realized).
- **Personalization/extraction**: as seller information rises from low→
  high, consumer surplus share falls from 84% to 68% (regression
  coefficient on "high info" = −0.160, p<0.001, clustered SE).
- **Search-driven concentration**: HHI of seller win-shares rises from
  ~0.10 (limited human search) to ~0.60 (full AI-mediated search) — but
  competitive multi-seller solicitation (M4) moderates this to ~0.45,
  a genuinely non-monotonic finding.
- **Buyer-side concentration (monopsony)**: common buyer-agent provider
  → higher consumer surplus share (0.80 vs 0.79 raw; regression
  significant at p=0.004 after controls). This is now a genuinely
  per-buyer mechanism (only the ~70% of buyers actually flagged as being
  on the shared provider receive the monopsony discount), which is why
  the gap is smaller than in the earlier, buggy market-wide-toggle version.
- **Seller-agent infrastructure concentration**: does NOT mechanically
  reduce aggregate market-wide price dispersion (in fact slightly raises
  it, because a market with a few large but *distinct* providers can be
  as heterogeneous as one with many independent sellers) — but shared-
  provider membership explains ~45% of the cross-seller variance in
  average markup within those markets, i.e. real within-network price
  correlation exists even though it doesn't show up as reduced
  market-wide dispersion. This is a nuanced point worth stating carefully
  in the discussion, not overclaiming "AI causes collusion."

### Data integrity fix (2026-08-19)
A code review of `code/simulate_market.py` flagged that `Buyer.common_provider`
(a per-buyer flag, drawn true for ~70% of buyers when
`buyer_concentration="common_buyer_provider"`) was generated but never
read anywhere in `negotiate()`/`run_transaction()`; monopsony power was
instead applied as one flat market-wide constant to *every* buyer in that
condition, so the intended 70%/30% per-buyer heterogeneity had zero
effect on any reported number. Fixed in `run_experiment()` so
`monopsony_power` is now computed per buyer from their own
`common_provider` flag (also added to the output CSV as
`buyer_common_provider` for transparency), and the full dataset/tables/
regressions/figures were regenerated from the fixed code. Two other
review findings were left as-is by design: hardcoded absolute paths from
the original build sandbox were fixed to relative paths (mechanical,
does not change any numbers), while a handful of minor stylistic items
(a dead unused column in `analysis.py`, some hardcoded figure y-axis
limits, duplicated HHI-formula code between `analysis.py` and
`make_figures.py`) were left for a later pass since they don't affect
correctness.

### 5. LLM-agent negotiation pilot (`code/llm_clients.py`, `code/llm_negotiation.py`, `code/analyze_llm_negotiation.py`)
Added 2026-08-19: a real (non-rule-based) validation layer, closing the
gap flagged in the original "future work" note below. Real Gemini
(`gemini-flash-lite-latest`, currently resolving to `gemini-3.5-flash-lite`
— the requested `gemini-3.6-flash-lite` does not exist as a model id) and
free OpenRouter models (`openrouter/free` auto-router with a named
fallback list, credentials rotated across two OpenRouter keys) play
buyer-agent and seller-agent roles and negotiate over a JSON action
protocol (`propose` / `accept` / `walk`), across the same
valuation/cost/bundle-value distributions as `simulate_market.py`.

Two regimes (`price_only`, `multiattr` — analogs of M2/M3) crossed with
three model pairings (gemini-gemini, openrouter-openrouter, and a mixed
gemini-buyer/openrouter-seller pairing) x 6 paired replicate buyer/seller
draws = 36 episodes. Outputs: `data/llm_negotiation_results.csv` (one row
per episode), `data/llm_negotiation_transcripts.jsonl` (full dialogues),
`data/table6_llm_validation.csv`, `data/llm_validation_summary.json`.

**This is an exploratory pilot, not a powered experiment**: real free-tier
LLM API calls are slow and occasionally fail to return parseable JSON (6
of 36 episodes here); every failure is logged and zero-filled, never
silently dropped. Because failures landed disproportionately in the
`multiattr` cell, the *unconditional* mean-welfare comparison is
confounded (multiattr looks worse only because more of its episodes
failed to close). The *deal-conditional* comparison
(`welfare_gain_multiattr_vs_price_only_llm_given_deal` in
`llm_validation_summary.json`) isolates surplus-creation-when-successful
and agrees directionally with the rule-based M3 > M2 finding
(+30.4 welfare units here vs. +13.4 in the rule-based sim). Treat this
as a qualitative robustness check to cite in the manuscript's
methodology/limitations section, not as a replacement for the
rule-based engine, which remains the main empirical source for all
headline numbers.

API keys (Gemini + 2x OpenRouter) live in a local, git-ignored `.env`
file — regenerate it locally if missing; never commit it.

```bash
# regenerate the LLM pilot (real API calls, ~15-20 min, needs .env)
python3 code/llm_negotiation.py
python3 code/analyze_llm_negotiation.py
```

## What is NOT done yet — this is the main remaining work

**The LaTeX manuscript itself has not been written yet.** Everything
above is the empirical/methodological engine; the actual paper (Elsevier
`elsarticle` class, ~30–40 pages) still needs to be drafted:

1. Title page, abstract (≤250 words), keywords (1–7), highlights (3–5
   bullets, ≤85 chars each) — per ECRA author guidelines (see the
   original conversation for the full guideline text if needed).
2. Introduction (the draft opening in the earlier conversation — the
   "customer wants to buy a television" narrative — is a good starting
   point and can likely be reused/adapted).
3. Literature review organized around: (a) e-commerce search/recommendation,
   (b) agentic commerce/markets (2025-2026), (c) algorithmic pricing and
   collusion, (d) personalization and price discrimination, (e)
   negotiation theory. Use `tex/references.bib`.
4. Formal model section: buyer utility U = v + bundle − price − delivery
   − risk; seller profit Π = price − cost − bundle_cost; regimes M0–M4 as
   institutional treatments; hypotheses H1–H7 (see the brainstorm in the
   earlier conversation for the hypothesis list — they map directly onto
   what the simulation actually tests).
5. **Methodology section must be honest about scope**: the MAIN engine is
   a rule-based ACE simulation (cite Tesfatsion; Faratin-Sierra-Jennings-
   style concession negotiation); explain why that's a legitimate and
   informative approach on its own (institutional/mechanism-level
   questions, full experimental control over valuations/costs/information,
   reproducibility). A `code/llm_negotiation.py` pilot with real Gemini +
   OpenRouter agents now exists (see item 5 above under "What's DONE") and
   should be reported as a small, exploratory robustness check corroborating
   the M2→M3 welfare-gain direction — not as the paper's main empirical
   claim (N=36, free-tier API noise, not a powered experiment).
6. Results section: walk through the six bullets above, citing
   `table1`–`table5`, `table_hhi.csv`, `table6_llm_validation.csv`, the
   regression outputs, and figures 1–4.
7. Discussion: tie back to Bichler (2026)'s "gains are not automatic"
   argument, the Coasean Singularity chapter, and algorithmic-collusion
   literature for the infrastructure-concentration result.
8. Limitations: rule-based agents as the central engine (LLM pilot is
   corroborating, small-N, not a replacement); stylized cost/valuation
   distributions; no real transaction data; English-language/single-
   product-category framing.
9. Declaration of Generative AI use (ECRA requires this — Claude was
   used substantively here, so this needs a real, honest disclosure
   statement, not the boilerplate "grammar check" exception).
10. CRediT author statement, competing interests, funding statement,
    data availability statement (the CSV + code IS the dataset — ECRA
    requires deposit in a repository or an explicit statement).
11. Compile with `pdflatex` (already confirmed to work in the sandbox —
    `elsarticle.cls` is installed via `texlive-publishers`) + `bibtex`,
    iterate until clean, produce final PDF.

## How to continue locally

```bash
# regenerate the dataset (deterministic, ~10s)
cd code && python3 simulate_market.py

# regenerate all tables/regressions
python3 analysis.py

# regenerate figures
python3 make_figures.py
```

Requires: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`
(`pip install numpy pandas scipy matplotlib statsmodels`).

For the LLM negotiation pilot, additionally: `requests`, `python-dotenv`,
`tenacity`, and a local `.env` with `GEMINI_API_KEY` / `OPENROUTER_API_KEY`
(/ `OPENROUTER_API_KEY_2`, etc. for key rotation).

For the LaTeX build, you'll need a TeX distribution with `elsarticle.cls`
(TeX Live: `texlive-publishers` package, or MiKTeX equivalent) plus
`natbib`.

## Files in this archive

```
code/
  simulate_market.py           # the ACE simulation engine
  analysis.py                  # all statistics/regressions
  make_figures.py              # the 4 figures
  llm_clients.py                # Gemini + OpenRouter API wrappers (retry/rotation)
  llm_negotiation.py             # real LLM buyer/seller negotiation pilot
  analyze_llm_negotiation.py      # summarizes the pilot vs. rule-based M2/M3
data/
  agentic_market_transactions.csv   # 105,600-row generated dataset
  headline_numbers.json             # every number for the prose
  table1..5_*.csv, table_hhi.csv    # summary tables
  reg_*.txt                         # regression outputs
  analysis_full_log.txt             # full console log of analysis.py
  llm_negotiation_results.csv        # one row per LLM-pilot episode
  llm_negotiation_transcripts.jsonl   # full LLM-pilot dialogues
  table6_llm_validation.csv           # LLM-pilot summary by regime x pairing
  llm_validation_summary.json         # LLM-pilot vs. rule-based comparison
figures/
  fig1..4_*.pdf   # vector figures for LaTeX \includegraphics
  fig1..4_*-1.png # PNG previews for quick viewing
tex/
  references.bib   # ~28 verified references with DOIs
README.md          # this file
```

# P2-ETF-HHMM-REGIME

**Hierarchical Hidden Markov Model for Multi‑Scale Regime Detection & ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-HHMM-REGIME/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HHMM-REGIME/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--hhmm--regime--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hhmm-regime-results)

## Overview

`P2-ETF-HHMM-REGIME` uses a **Hierarchical Hidden Markov Model** to detect multi‑scale market regimes:
- **Macro regime** (e.g., risk‑on vs. risk‑off)
- **Sector regime** (e.g., growth vs. value leadership)
- **ETF‑specific regime** (e.g., trending vs. mean‑reverting)

The engine selects top ETFs based on recent returns, contextualized by the current regime state.

## Methodology

1. **Top‑Level HMM**: Fit on macro features or PCA of returns (3 states).
2. **Mid‑Level HMM**: Fit per macro state on ETF returns (2 states).
3. **Bottom‑Level HMM**: Fit per (macro, sector) on mean ETF returns (2 states).
4. **Regime Prediction**: Predict current state at all three levels.
5. **ETF Ranking**: Rank ETFs by 21‑day annualized return.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
text

The HHMM‑REGIME engine is complete and ready for deployment. It adds hierarchical regime awareness to your suite, filling the multi‑scale gap.

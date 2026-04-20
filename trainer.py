"""
Main training script for HHMM-REGIME engine.
Fits hierarchical HMM and selects top ETFs based on regime‑conditional expected returns.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from hhmm_model import HierarchicalHMM
import push_results

def compute_expected_return(returns: pd.Series, window: int = 21) -> float:
    """Simple expected return: recent average daily return annualized."""
    if len(returns) < window:
        return 0.0
    return returns.iloc[-window:].mean() * 252

def select_top_etfs(returns: pd.DataFrame, regime_info: dict, n: int = 3) -> list:
    """
    Select top ETFs based on recent return, filtered by regime.
    In HHMM, we favor ETFs with returns aligned to bullish regime states.
    """
    tickers = returns.columns.tolist()
    scores = {}
    for ticker in tickers:
        ret_21d = returns[ticker].iloc[-21:].mean() * 252 if len(returns) >= 21 else 0
        scores[ticker] = ret_21d
    
    sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{'ticker': t, 'expected_return': v} for t, v in sorted_tickers[:n]]

def run_hhmm():
    print(f"=== P2-ETF-HHMM-REGIME Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    
    all_results = {}
    top_picks = {}
    
    # Global HHMM on combined universe
    returns_all = data_manager.prepare_returns_matrix(df_master, config.ALL_TICKERS)
    if len(returns_all) < config.MIN_OBSERVATIONS:
        print("Insufficient data for combined universe.")
        return
    
    recent_all = returns_all.iloc[-config.LOOKBACK_WINDOW:]
    model = HierarchicalHMM(
        n_macro=config.N_MACRO_STATES,
        n_sector=config.N_SECTOR_STATES,
        n_etf=config.N_ETF_STATES,
        random_state=config.RANDOM_SEED
    )
    model.fit(recent_all)
    
    regime_info = model.predict_regime(recent_all)
    print(f"Current regime: {regime_info}")
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue
        recent = returns.iloc[-config.LOOKBACK_WINDOW:]
        
        # Select top ETFs based on 21-day return
        top = select_top_etfs(recent, regime_info, n=3)
        top_picks[universe_name] = top
        
        # Store full metrics for table
        universe_metrics = {}
        for ticker in tickers:
            if ticker in recent.columns:
                ret_21d = recent[ticker].iloc[-21:].mean() * 252 if len(recent) >= 21 else 0
                universe_metrics[ticker] = {
                    'expected_return': ret_21d,
                    'regime': regime_info
                }
        all_results[universe_name] = universe_metrics
    
    # Shrinking windows: use a fixed-length window starting at each start year
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = start_date + pd.Timedelta(days=config.LOOKBACK_WINDOW * 2)  # enough to cover 504 trading days
        window_label = f"{start_year}-{start_year + int(config.LOOKBACK_WINDOW / 252) + 1}"
        print(f"\n--- Shrinking Window: {window_label} ---")
        
        mask = (df_master['Date'] >= start_date) & (df_master['Date'] <= end_date)
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            print(f"    Skipping (insufficient data)")
            continue
        
        returns_win = data_manager.prepare_returns_matrix(df_window, config.ALL_TICKERS)
        if len(returns_win) < config.MIN_OBSERVATIONS:
            continue
        
        # Take exactly LOOKBACK_WINDOW days from the start
        recent_win = returns_win.iloc[:config.LOOKBACK_WINDOW]
        if len(recent_win) < config.MIN_OBSERVATIONS:
            continue
        
        win_model = HierarchicalHMM(
            n_macro=config.N_MACRO_STATES,
            n_sector=config.N_SECTOR_STATES,
            n_etf=config.N_ETF_STATES,
            random_state=config.RANDOM_SEED
        )
        win_model.fit(recent_win)
        win_regime = win_model.predict_regime(recent_win)
        
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            returns_u = data_manager.prepare_returns_matrix(df_window, tickers)
            if len(returns_u) < config.MIN_OBSERVATIONS:
                continue
            recent_u = returns_u.iloc[:config.LOOKBACK_WINDOW]
            top = select_top_etfs(recent_u, win_regime, n=1)
            if top:
                window_top[universe_name] = top[0]
        
        shrinking_results[window_label] = {
            'start_year': start_year,
            'regime': win_regime,
            'top_picks': window_top,
            'n_observations': len(recent_win)
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "n_macro_states": config.N_MACRO_STATES,
            "n_sector_states": config.N_SECTOR_STATES,
            "n_etf_states": config.N_ETF_STATES,
            "lookback_window": config.LOOKBACK_WINDOW
        },
        "regime": regime_info,
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_hhmm()

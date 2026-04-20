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
    
    win_model = VineCopulaModel(margin_model=config.MARGIN_MODEL)
    if not win_model.fit(recent_win):
        continue
    
    win_sim = win_model.simulate(n_sim=config.N_SIMULATIONS // 2)
    win_metrics = win_model.compute_risk_metrics(win_sim)
    
    window_top = {}
    for universe_name, tickers in config.UNIVERSES.items():
        best_ticker = max(
            [t for t in tickers if t in win_metrics],
            key=lambda t: win_metrics[t]['expected_return'],
            default=None
        )
        if best_ticker:
            window_top[universe_name] = {
                'ticker': best_ticker,
                'expected_return': win_metrics[best_ticker]['expected_return'],
                'combined_score': win_metrics[best_ticker]['combined_score']
            }
    
    shrinking_results[window_label] = {
        'start_year': start_year,
        'top_picks': window_top,
        'n_observations': len(recent_win)
    }

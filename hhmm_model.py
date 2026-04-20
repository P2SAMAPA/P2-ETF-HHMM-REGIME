"""
Hierarchical Hidden Markov Model for multi‑scale regime detection.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class HierarchicalHMM:
    def __init__(self, n_macro=3, n_sector=2, n_etf=2, random_state=42):
        self.n_macro = n_macro
        self.n_sector = n_sector
        self.n_etf = n_etf
        self.random_state = random_state
        
        self.macro_model = None
        self.sector_models = {}      # macro_state -> dict of sector models
        self.etf_models = {}         # (macro_state, sector_state) -> dict of ETF models
        
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, returns: pd.DataFrame, macro_features: pd.DataFrame = None):
        """
        Fit HHMM to multivariate returns.
        If macro_features provided, use them for top level; otherwise PCA of returns.
        """
        self.tickers = returns.columns.tolist()
        X = self.scaler.fit_transform(returns.values)
        
        # --- Top level: Macro regime ---
        # Use either macro features or first principal component of returns
        if macro_features is not None and len(macro_features) > 0:
            macro_data = macro_features.values
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            macro_data = pca.fit_transform(X)
        
        self.macro_model = hmm.GaussianHMM(
            n_components=self.n_macro,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=100
        )
        self.macro_model.fit(macro_data)
        macro_states = self.macro_model.predict(macro_data)
        
        # --- Mid level: Sector regime per macro state ---
        for m in range(self.n_macro):
            mask = macro_states == m
            if np.sum(mask) < 50:
                continue
            sector_model = hmm.GaussianHMM(
                n_components=self.n_sector,
                covariance_type="diag",
                random_state=self.random_state,
                n_iter=50
            )
            sector_model.fit(X[mask])
            self.sector_models[m] = sector_model
        
        # --- Bottom level: ETF regime per (macro, sector) state ---
        for m in range(self.n_macro):
            if m not in self.sector_models:
                continue
            mask_m = macro_states == m
            sector_states = self.sector_models[m].predict(X[mask_m])
            
            for s in range(self.n_sector):
                mask_s = sector_states == s
                if np.sum(mask_s) < 20:
                    continue
                # Fit a simple 2-state HMM on the mean return across ETFs
                etf_data = X[mask_m][mask_s].mean(axis=1, keepdims=True)
                if len(etf_data) < 20:
                    continue
                etf_model = hmm.GaussianHMM(
                    n_components=self.n_etf,
                    covariance_type="diag",
                    random_state=self.random_state,
                    n_iter=30
                )
                etf_model.fit(etf_data)
                self.etf_models[(m, s)] = etf_model
        
        self.fitted = True
        return True
    
    def predict_regime(self, returns: pd.DataFrame, macro_features: pd.DataFrame = None) -> dict:
        """Predict the current regime state for each level."""
        if not self.fitted:
            return {}
        
        X = self.scaler.transform(returns.values)
        
        if macro_features is not None and len(macro_features) > 0:
            macro_data = macro_features.values[-1:].reshape(1, -1)
        else:
            macro_data = X[-1:].mean(axis=1, keepdims=True)
        
        macro_state = self.macro_model.predict(macro_data)[0]
        macro_prob = np.max(self.macro_model.predict_proba(macro_data)[0])
        
        result = {
            'macro_state': int(macro_state),
            'macro_prob': float(macro_prob)
        }
        
        if macro_state in self.sector_models:
            sector_model = self.sector_models[macro_state]
            sector_state = sector_model.predict(X[-1:])[0]
            sector_prob = np.max(sector_model.predict_proba(X[-1:])[0])
            result['sector_state'] = int(sector_state)
            result['sector_prob'] = float(sector_prob)
            
            key = (macro_state, sector_state)
            if key in self.etf_models:
                etf_model = self.etf_models[key]
                etf_data = X[-1:].mean(axis=1, keepdims=True)
                etf_state = etf_model.predict(etf_data)[0]
                etf_prob = np.max(etf_model.predict_proba(etf_data)[0])
                result['etf_state'] = int(etf_state)
                result['etf_prob'] = float(etf_prob)
        
        return result

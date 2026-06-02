import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. DATA GENERATION ---

class StochasticProcess:
    def __init__(self, S0, mu, sigma, T, N):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N
        self.t = np.linspace(0, T, N)
    
    def geometric_brownian_motion(self):
        W = np.random.standard_normal(size=self.N)
        W = np.cumsum(W) * np.sqrt(self.dt)
        X = (self.mu - 0.5 * self.sigma**2) * self.t + self.sigma * W
        S = self.S0 * np.exp(X)
        S[0] = self.S0
        return S

# --- 2. SIMULATION ENGINE ---

class MonteCarloSimulation:
    def __init__(self, fee):
        self.fee = fee 

    def run_fixed_path(self, prices_df, strategy):
        initial_value = 1_000_000.0
        if self.fee == 0.0:
            wx, wy = strategy.start_weights[0], strategy.start_weights[1]
            P_array = prices_df["Asset A"].values
            P0 = P_array[0]

            x0 = (initial_value * wx) / P0 
            y0 = (initial_value * wy) / 1.0

            L = (x0 ** wx) * (y0 ** wy) 

            x_star = L * (wx / (P_array * wy)) ** wy
            y_star = L * ((P_array * wy) / wx) ** wx
            portfolio_values = (P_array * x_star) + y_star
            return portfolio_values
        else:
            return np.array([])

class RebalancingStrategy:
    def __init__(self, mode, start_weights, target_weights, steps):
        self.mode = mode
        self.start_weights = np.array(start_weights)
        self.target_weights = np.array(target_weights)
        self.steps = steps

# --- 3. EXHAUSTIVE PARAMETRIC SWEEP EXPERIMENT ---

def run_exhaustive_parametric_sweep(trials=10000):
    print(f"Initializing Continuous Stochastic Parameter Sweep across {trials} independent configurations...")
    steps = 253
    T = 1.0
    initial_price = 100.0
    initial_capital = 1_000_000.0
    sim = MonteCarloSimulation(fee=0.0)

    # Storage for empirical data collection
    sweep_records = []

    for path in tqdm(range(trials)):
        # Generate entirely randomized continuous parameters for this path
        # w_A sweeps continuously up to a maximum 50/50 split (0.50)
        w_A = np.random.uniform(0.01, 0.50)
        w_C = 1.0 - w_A
        
        # Drift (mu) sweeps continuously from a 2% baseline up to a 50% hyper-growth surge
        mu = np.random.uniform(0.02, 0.50)
        
        # Volatility (sigma) sweeps continuously from a stable 5% up to a chaotic 70%
        sigma = np.random.uniform(0.05, 0.70)
        
        strategy = RebalancingStrategy('static', [w_A, w_C], [w_A, w_C], steps)
        
        # Generate the unique path based on these random coordinates
        proc = StochasticProcess(S0=initial_price, mu=mu, sigma=sigma, T=T, N=steps)
        path_a = proc.geometric_brownian_motion()
        prices_df = pd.DataFrame({"Asset A": path_a, "Cash": np.ones(steps)})
        
        # Execute the rebalancing simulation
        val_history = sim.run_fixed_path(prices_df, strategy)
        
        # Capture terminal boundary values
        final_asset_price = path_a[-1]
        final_portfolio_value = val_history[-1]
        
        realized_amm_return = ((final_portfolio_value / initial_capital) - 1.0) * 100.0
        realized_asset_return = ((final_asset_price / initial_price) - 1.0) * 100.0
        
        # Calculate theoretical continuous geometric expectation
        theoretical_geo_target = (mu * wA) - (0.5 * sigma**2 * wA**2) if 'wA' in locals() else (mu * w_A) - (0.5 * sigma**2 * w_A**2)
        
        # Map the structural tracking error for this specific unique timeline
        theoretical_terminal_value = initial_capital * (final_asset_price / initial_price)**w_A * np.exp(0.5 * w_A * w_C * sigma**2 * T)
        absolute_tracking_divergence = np.abs(final_portfolio_value - theoretical_terminal_value)

        sweep_records.append({
            "Path_ID": path + 1,
            "Weight_Asset_A": w_A,
            "Weight_Cash": w_C,
            "Drift_Mu": mu,
            "Volatility_Sigma": sigma,
            "Asset_Terminal_Price": final_asset_price,
            "Portfolio_Terminal_Value": final_portfolio_value,
            "Realized_AMM_Yield_Pct": realized_amm_return,
            "Realized_Asset_Yield_Pct": realized_asset_return,
            "Tracking_Divergence_USD": absolute_tracking_divergence
        })

    # Compile dataset into a structural ledger
    df_sweep = pd.DataFrame(sweep_records)
    
    # Save directly to disk for portfolio modeling software
    df_sweep.to_csv("amm_continuous_parametric_sweep.csv", index=False)
    print("\n" + "="*80)
    print("                 CONTINUOUS PARAMETRIC SWEEP STATISTICAL LEDGER")
    print("="*80)
    print(df_sweep.head(20).to_string(index=False))
    print("...\n[Remaining records compiled in 'amm_continuous_parametric_sweep.csv']")
    print("="*80)

if __name__ == "__main__":
    run_exhaustive_parametric_sweep(trials=1000)

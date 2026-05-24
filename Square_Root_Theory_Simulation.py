import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# --- 2. AMM MATH & PRICING ---

class AMMPricing:
    @staticmethod
    def calc_bid_amm(xi, xj, wi, wj, fee):
        numerator = (1 - fee) * xj * wi
        denominator = (wj + fee * wi) * xi
        return numerator / denominator
    
    @staticmethod
    def calc_external_bid(Pia, Pjb):
        return Pia / Pjb
    
    @staticmethod
    def get_delta_j(xi, xj, wi, wj, delta_i, fee):
        frac = xi / (xi + delta_i)
        exponent = ((1 - fee) * wi) / wj
        return xj * (frac ** exponent - 1)

# --- 3. ARBITRAGE LOGIC ---

class ArbitrageDetector:
    def __init__(self, fee):
        self.fee = fee
        self.pricing = AMMPricing()
    
    def check_arb_opportunity(self, reserves, prices, weights, mid_prices, bid_ask):
        total_value = np.sum(mid_prices * reserves)
        realized_weights = (mid_prices * reserves) / total_value
        discrepancies = realized_weights - weights
        over_weighted_index = np.argmax(discrepancies)
        under_weighted_index = np.argmin(discrepancies)
        
        if bid_ask:
            external_implied = self.pricing.calc_external_bid(
                prices[under_weighted_index][1], # Ask
                prices[over_weighted_index][0]   # Bid
            )
        else:
            external_implied = self.pricing.calc_external_bid(
                prices[under_weighted_index], 
                prices[over_weighted_index]
            )
        
        amm_implied = self.pricing.calc_bid_amm(
            reserves[under_weighted_index],
            reserves[over_weighted_index],
            weights[under_weighted_index],
            weights[over_weighted_index],
            self.fee
        )
        return external_implied, amm_implied, over_weighted_index, under_weighted_index

class NewtonSolver:
    def __init__(self, fee, tol=1e-12, max_iter=100):
        self.fee = fee
        self.tol = tol
        self.max_iter = max_iter
        self.pricing = AMMPricing()
    
    def _f(self, xi, xj, wi, wj, Pia, Pjb, delta_i):
        dj = self.pricing.get_delta_j(xi, xj, wi, wj, delta_i, self.fee)
        amm_bid = self.pricing.calc_bid_amm(xi + delta_i, xj + dj, wi, wj, self.fee)
        P_star = self.pricing.calc_external_bid(Pia, Pjb)
        return amm_bid - P_star
    
    def _bisect(self, obj, lo, hi):
        for _ in range(self.max_iter):
            mid = (lo + hi) / 2
            if obj(mid) * obj(lo) > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < self.tol:
                return mid
        return (lo + hi) / 2

    def solve_delta_i(self, xi, xj, wi, wj, Pia, Pjb, initial_guess):
        current_price = self.pricing.calc_bid_amm(xi, xj, wi, wj, self.fee)
        target_price = self.pricing.calc_external_bid(Pia, Pjb)
        if target_price > current_price: return 0.0, 0.0
        try:
            delta_i = self._bisect(lambda x: self._f(xi, xj, wi, wj, Pia, Pjb, x), 0.0, xi * 0.5)
            delta_j = self.pricing.get_delta_j(xi, xj, wi, wj, delta_i, self.fee)
            return delta_i, delta_j
        except:
            return 0.0, 0.0

class SequentialArbitrageSolver:
    def __init__(self, fee, tol=1e-12, max_iter=100):
        self.detector = ArbitrageDetector(fee)
        self.solver = NewtonSolver(fee, tol)
    
    def solve(self, reserves, prices, mid_prices, weights, bid_ask):
        reserves = np.array(reserves, dtype=float)
        weights = np.array(weights, dtype=float)
        for k in range(100):
            ext, amm, over, under = self.detector.check_arb_opportunity(reserves, prices, weights, mid_prices, bid_ask)
            if abs(ext - amm) < 1e-4 or amm <= ext: break
            i, wi, j, wj = reserves[under], weights[under], reserves[over], weights[over]
            Pia, Pjb = (prices[under][1], prices[over][0]) if bid_ask else (prices[under], prices[over])
            delta_i, delta_j = self.solver.solve_delta_i(i, j, wi, wj, Pia, Pjb, i*0.01)
            if delta_i <= 0: break
            reserves[under] += delta_i
            reserves[over] += delta_j
        return reserves

# --- 4. SIMULATION ENGINE ---

class MonteCarloSimulation:
    def __init__(self, fee):
        self.fee = fee #This is for the if statement to follow 
        self.solver = SequentialArbitrageSolver(fee)

    def run_fixed_path(self, prices_df, strategy):
        initial_value = 1_000_000
        #---New Branching Logic---
        if self.fee == 0.0: #setup statment for the closed-form solutions
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
            
            reserves = None
            portfolio_values = []
            for t, prices in enumerate(prices_df.values):
                current_target = strategy.get_weights_at_step(t)
                if t == 0:
                    initial_value = 1_000_000
                    reserves = (initial_value * strategy.start_weights) / prices
                reserves = self.solver.solve(
                    reserves=reserves,
                    prices=np.array([(p*0.999, p*1.001) for p in prices]), # Minimal spread
                    mid_prices=prices,
                    weights=current_target,
                    bid_ask=True
                )
                portfolio_values.append(np.sum(prices * reserves))
            return np.array(portfolio_values)



class RebalancingStrategy:
    def __init__(self, mode, start_weights, target_weights, steps):
        self.mode = mode
        self.start_weights = np.array(start_weights)
        self.target_weights = np.array(target_weights)
        self.steps = steps

    def get_weights_at_step(self, t):
        return self.start_weights # Static 50/50 for this demo

# --- 5. THE CLEAN LEDGER EXPERIMENT ---

def run_square_root_simulation(trials):
    # --- 1. Parameters ---
    steps = 253
    initial_price = 100.0

    # --- 2. The Strategy ---
    theory_strat = RebalancingStrategy(
        mode='static', 
        start_weights=[0.5, 0.5], 
        target_weights=[0.5, 0.5], 
        steps=steps
    )
    
    # --- 4. Run Simulation ---
    sim = MonteCarloSimulation(fee=0.0) 
    terminal_prices = np.zeros(trials)
    terminal_values = np.zeros(trials)

    for i in tqdm(range(trials)):
        proc = StochasticProcess(S0=initial_price, mu=0.05, sigma=0.4, T=1.0, N=steps)
        path_a = proc.geometric_brownian_motion()
        prices_df = pd.DataFrame({"Asset A": path_a, "Cash": np.ones(steps)})

        val_history = sim.run_fixed_path(prices_df, theory_strat)
        
        terminal_prices[i] = path_a[-1]
        terminal_values[i] = val_history[-1]

    #---5. Statistical Output ---
    worst_idx = np.argmin(terminal_values)
    best_idx = np.argmax(terminal_values)
    
    print("-" * 55)
    print("THE ASSET (Geometric Brownian Motion)")
    print(f"Median Terminal Price A:      ${np.median(terminal_prices):,.2f}  <-- Most Likely Reality")
    print(f"Mean Terminal Price A:        ${np.mean(terminal_prices):,.2f}  <-- Skewed by extreme outliers")
    
    print("-" * 55)
    print("THE PORTFOLIO (AMM ETF)")
    print(f"Median Terminal Portfolio V:  ${np.median(terminal_values):,.2f}  <-- Expected Geometric Growth") 
    print(f"Mean Terminal Portfolio V:    ${np.mean(terminal_values):,.2f}  <-- Skewed by extreme outliers")
    
    print("-" * 55)
    print("EXTREMES (Out of 100,000 paths)")
    print(f"Worst-Case Portfolio:  ${terminal_values[worst_idx]:,.2f}  (Asset A Price was: ${terminal_prices[worst_idx]:,.2f})")
    print(f"Best-Case Portfolio:   ${terminal_values[best_idx]:,.2f}  (Asset A Price was: ${terminal_prices[best_idx]:,.2f})")
            
    
   

if __name__ == "__main__":
    run_square_root_simulation(100_000)

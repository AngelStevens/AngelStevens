import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from typing import List, Tuple, Dict, Optional

# --- 1. DATA GENERATION ---

class StochasticProcess:
    """
    Generates synthetic stock price data using Geometric Brownian Motion.
    """
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

    def exp_brownian_bridge(self,start,end):
        """Generates price path that ends at "end" price"""
        #Generates random walk
        W = np.random.standard_normal(size = self.N)
        W = np.cumsum(W)*np.sqrt(self.dt)
        #Turn it into a "Bridge"
        bridge = W - (self.t/self.T)*W[-1]
        #creates log-linear slop from start to end
        drift = np.log(start)+(self.t/self.T)*(np.log(end)-np.log(start))
        #combine the slope with random bridge
        X = drift+self.sigma*bridge
        S = np.exp(X)
        S[0] = start
        S[-1] = end
        return S

# --- 2. AMM MATH & PRICING ---

class AMMPricing:
    """
    Mathematical formulas for the Constant Mean Market Maker.
    """
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
        """
        YOUR SPECIFIC FORMULA:
        Calculates how much output (stock j) is given for input (stock i).
        Returns a negative value representing assets leaving the pool.
        """
        frac = xi / (xi + delta_i)
        exponent = ((1 - fee) * wi) / wj
        return xj * (frac ** exponent - 1)

# --- 3. ARBITRAGE LOGIC ---

class ArbitrageDetector:
    def __init__(self, fee):
        self.fee = fee
        self.pricing = AMMPricing()
    
    def check_arb_opportunity(self, reserves, prices, weights, mid_prices, bid_ask):
        # Calculate current realized weights
        total_value = np.sum(mid_prices * reserves)
        realized_weights = (mid_prices * reserves) / total_value
        
        # Find discrepancies
        discrepancies = realized_weights - weights
        over_weighted_index = np.argmax(discrepancies)
        under_weighted_index = np.argmin(discrepancies)
        
        # Compare Prices
        if bid_ask:
            # We buy the "cheap" asset from market (ask) and sell "expensive" to market (bid)
            external_implied = self.pricing.calc_external_bid(
                prices[under_weighted_index][1], # Ask price of underweighted
                prices[over_weighted_index][0]   # Bid price of overweighted
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
    """
    Solves for the exact trade size to close the arbitrage gap.
    """
    def __init__(self, fee, tol=1e-12, max_iter=100):
        self.fee = fee
        self.tol = tol
        self.max_iter = max_iter
        self.pricing = AMMPricing()
    
    def _f(self, xi, xj, wi, wj, Pia, Pjb, delta_i):
        # Objective: AMM Price after trade - External Price = 0
        dj = self.pricing.get_delta_j(xi, xj, wi, wj, delta_i, self.fee)
        amm_bid = self.pricing.calc_bid_amm(xi + delta_i, xj + dj, wi, wj, self.fee)
        P_star = self.pricing.calc_external_bid(Pia, Pjb)
        return amm_bid - P_star
    
    def _bisect(self, obj, lo, hi):
        # Simple bisection root finder
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
        # Check if arb exists first
        current_price = self.pricing.calc_bid_amm(xi, xj, wi, wj, self.fee)
        target_price = self.pricing.calc_external_bid(Pia, Pjb)
        
        if target_price > current_price: 
            return 0.0, 0.0 # No arb or wrong direction
            
        # Use Bisection to find optimal input amount
        # Upper bound: we assume we won't trade more than 50% of the pool in one go
        try:
            delta_i = self._bisect(
                lambda x: self._f(xi, xj, wi, wj, Pia, Pjb, x), 
                0.0, 
                xi * 0.5
            )
            delta_j = self.pricing.get_delta_j(xi, xj, wi, wj, delta_i, self.fee)
            return delta_i, delta_j
        except:
            return 0.0, 0.0

class SequentialArbitrageSolver:
    def __init__(self, fee, tol=1e-12, max_iter=100):
        self.fee = fee
        self.tol = tol
        self.max_iter = max_iter
        self.detector = ArbitrageDetector(fee)
        self.solver = NewtonSolver(fee, tol)
    
    def solve(self, reserves, prices, mid_prices, weights, bid_ask):
        reserves = np.array(reserves, dtype=float)
        weights = np.array(weights, dtype=float)
        
        # Realized weights
        realized_weights = (mid_prices * reserves) / np.sum(mid_prices * reserves)
        discrepancies = realized_weights - weights
        
        for k in range(self.max_iter):
            # Check for opportunity
            ext, amm, over, under = self.detector.check_arb_opportunity(
                reserves, prices, weights, mid_prices, bid_ask
            )
            
            # If AMM price is close enough to External, stop.
            if abs(ext - amm) < 1e-4 or amm <= ext:
                break
                
            # Execute Trade
            i, wi = reserves[under], weights[under] # Asset coming IN (underweighted)
            j, wj = reserves[over], weights[over]   # Asset going OUT (overweighted)
            
            # Get external prices
            if bid_ask:
                Pia = prices[under][1] # Ask
                Pjb = prices[over][0]  # Bid
            else:
                Pia, Pjb = prices[under], prices[over]
                
            delta_i, delta_j = self.solver.solve_delta_i(i, j, wi, wj, Pia, Pjb, i*0.01)
            
            if delta_i <= 0: break
            
            # Update Reserves
            reserves[under] += delta_i
            reserves[over] += delta_j # delta_j is negative
            
            # Update discrepancies for next loop
            realized_weights = (mid_prices * reserves) / np.sum(mid_prices * reserves)
            discrepancies = realized_weights - weights
            
        return reserves, discrepancies, k

# --- 4. SIMULATION ENGINE ---

class MonteCarloSimulation:
    def __init__(self, fee):
        self.fee = fee
        self.solver = SequentialArbitrageSolver(fee)
    
    def _generate_price_paths(self, portfolio_size, steps, lower_vol, upper_vol, initial_price=100.0):
        prices_dict = {}
        # Simple volatility selection
        vol = (lower_vol + upper_vol) / 2 
        
        for i in range(portfolio_size):
            process = StochasticProcess(S0=initial_price, mu=0, sigma=vol, T=1.0, N=steps)
            prices_dict[f"Asset {i}"] = process.geometric_brownian_motion()
        return pd.DataFrame(prices_dict)

    def run_trial(self, steps, portfolio_size, start_weights, target_weights, lower_vol, upper_vol, initial_value=1_000_000):
        # 1. Generate Price Data
        prices_df = self._generate_price_paths(portfolio_size, steps, lower_vol, upper_vol)
        daily_prices = [row.values for _, row in prices_df.iterrows()]
        
        portfolio_values = []
        weight_discrepancies = []
        
        # 2. Simulation Loop
        for t, prices in enumerate(daily_prices):
            
            # A. Calculate Interpolated Weights (The "Moving Target")
            progress = t / steps
            if progress > 1.0: progress = 1.0
            current_target_weights = start_weights + (target_weights - start_weights) * progress
            
            # B. Prepare Prices (Spread = 1%)
            spread = 0.01
            prices_with_spread = np.array([(p*(1-spread), p*(1+spread)) for p in prices])
            mid_prices = prices
            
            # C. Initialize Reserves (Step 0)
            if t == 0:
                reserves = (initial_value * start_weights) / mid_prices
            
            # D. Arbitrage (Rebalancing)
            try:
                reserves, discrepancies, _ = self.solver.solve(
                    reserves=reserves,
                    prices=prices_with_spread,
                    mid_prices=mid_prices,
                    weights=current_target_weights,
                    bid_ask=True
                )
            except Exception as e:
                pass # Skip if solver fails
            
            # E. Track Metrics
            portfolio_values.append(np.sum(mid_prices * reserves))
            
            # Calculate error against the *current* target
            realized = (mid_prices * reserves) / np.sum(mid_prices * reserves)
            weight_discrepancies.append(realized - current_target_weights)
            
        return np.array(portfolio_values), weight_discrepancies

    def run_fixed_path(self,prices_df,strategy):
        reserves = None
        portfolio_values = []
        for t,prices in enumerate(prices_df.values):
            current_target = strategy.get_weights_at_step(t)
            if t == 0:
                initial_value = 1_000_000
                reserves = (initial_value * strategy.start_weights) / prices
            reserves, tracking_error, solver_steps = self.solver.solve(
            reserves=reserves,
            prices=np.array([(p*0.99, p*1.01) for p in prices]), # Adding spread
            mid_prices=prices,
            weights=current_target,
            bid_ask=True
        )
            portfolio_values.append(np.sum(prices * reserves))
        return np.array(portfolio_values)

class ExperimentRunner:
    def __init__(self):
        self.results = {"Portfolio Vals": [], "Weight Discrepancies": []}
    
    def run_experiment(self, trials=10, steps=252, portfolio_size=2):
        print(f"Running {trials} trials with {steps} steps...")
        
        for _ in tqdm(range(trials)):
            # Define Start and End Weights
            # Scenario: Shift from 50/50 to 20/80
            start_weights = np.array([0.5, 0.5])
            target_weights = np.array([0.2, 0.8])
            
            sim = MonteCarloSimulation(fee=0.003)
            p_vals, w_disc = sim.run_trial(
                steps=steps, 
                portfolio_size=portfolio_size, 
                start_weights=start_weights, 
                target_weights=target_weights,
                lower_vol=0.5, upper_vol=0.5
            )
            
           # Store results:
            # 1. Normalize portfolio value to percentage (Start = 100%)
            self.results["Portfolio Vals"].append((p_vals / p_vals[0]) * 100)
            
            # 2. Store the weight error (Mean Absolute Error for this run)
            # We take the average error across all time steps for this specific trial
            avg_error = np.mean(np.abs(w_disc))
            self.results["Weight Discrepancies"].append(avg_error)
            
    def print_summary_statistics(self):
        # 1. Get the FINAL value of the portfolio for every trial
        final_values = [run[-1] for run in self.results["Portfolio Vals"]]
        
        # 2. Calculate Stats
        mean_val = np.mean(final_values)
        std_dev = np.std(final_values)
        mean_error = np.mean(self.results["Weight Discrepancies"])
        
        print("\n" + "="*40)
        print("       SIMULATION RESULTS SUMMARY       ")
        print("="*40)
        print(f"Total Trials:          {len(final_values)}")
        print(f"Mean Final Value:      {mean_val:.2f}%")
        print(f"Standard Deviation:    {std_dev:.2f}%")
        print(f"Sharpe Ratio (est):    {(mean_val - 100) / std_dev:.4f}")
        print(f"Avg Tracking Error:    {mean_error:.6f}")
        print("-" * 40)
        print("Interpretation:")
        print(" > Mean > 100% means the rebalancing + fees generated profit.")
        print(" > Low Tracking Error means the bot followed the target weights closely.")
        print("="*40 + "\n")
            
    def plot_results(self):
        plt.figure(figsize=(10, 6))
        for run in self.results["Portfolio Vals"]:
            plt.plot(run, alpha=0.3, color='blue')
        plt.title("Portfolio Rebalancing Performance (50/50 -> 20/80)")
        plt.xlabel("Time Steps")
        plt.ylabel("Portfolio Value (%)")
        plt.grid(True, alpha=0.3)
        plt.show()

class RebalancingStrategy:
    def __init__(self,mode,start_weights,target_weights,steps):
        self.mode = mode
        self.start_weights = np.array(start_weights)
        self.target_weights = np.array(target_weights)
        self.steps = steps

    def get_weights_at_step(self, t):
        if self.mode == "static":
            return self.start_weights
        if self.mode == "immediate":
            return self.target_weights if t>0 else self.start_weights
        progress = min(t/self.steps, 1.0)
        return self.start_weights+(self.target_weights - self.start_weights)*progress


def price_path_comparison():
    # 1. Setup the "Fixed Path" (The Lead's request)
    steps = 200
    proc = StochasticProcess(S0=100, mu=0.05, sigma=0.2, T=1.0, N=steps)
    p0 = proc.geometric_brownian_motion()
    p1 = proc.geometric_brownian_motion()
    fixed_prices = pd.DataFrame({"Asset 0": p0, "Asset 1": p1})

    # 2. Initialize the Simulation engine
    sim = MonteCarloSimulation(fee=0.003)
    
    # 3. Create the two "Competitors"
    grad_strat = RebalancingStrategy('gradual', [0.5, 0.5], [0.8, 0.2], steps)
    imm_strat = RebalancingStrategy('immediate', [0.5, 0.5], [0.8, 0.2], steps)

    # 4. Run them on the same path
    val_gradual = sim.run_fixed_path(fixed_prices, grad_strat)
    val_immediate = sim.run_fixed_path(fixed_prices, imm_strat)

    # Calculate the exact dollar difference
    value_difference = val_gradual - val_immediate

    # 5. Plot the showdown (Two Subplots)
    plt.figure(figsize=(10, 8))

    # Top Chart: The Macro View
    plt.subplot(2, 1, 1)
    plt.plot(val_gradual, label="Gradual (Linear)", color='green', linewidth=3)
    # Using a dashed line so the green shows through!
    plt.plot(val_immediate, label="Immediate (Shock)", color='red', linestyle='--') 
    plt.title("Macro View: Overall Portfolio Value")
    plt.ylabel("Value ($)")
    plt.legend()

    # Bottom Chart: The "Alpha" View
    plt.subplot(2, 1, 2)
    plt.plot(value_difference, label="Gradual Edge ($ Saved)", color='blue', linewidth=2)
    plt.title("The 'Alpha' (Gradual Value minus Immediate Value)")
    plt.xlabel("Time Step")
    plt.ylabel("Difference ($)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def bridge_path_comparison():
    # 1. Setup the "Rigged Market" (Brownian Bridge)
    steps = 200
    proc = StochasticProcess(S0=100, mu=0.0, sigma=0.2, T=1.0, N=steps)

    # Asset 0 is the "Loser" -> Drops to 80
    p0 = proc.exp_brownian_bridge(start=100, end=80)
    # Asset 1 is the "Winner" -> Rises to 120
    p1 = proc.exp_brownian_bridge(start=100, end=120)
    
    fixed_prices = pd.DataFrame({"Asset 0": p0, "Asset 1": p1})

    # 2. Initialize the Simulation engine
    sim = MonteCarloSimulation(fee=0.003)
    
    # 3. Create the two "Competitors"
    # Target Weights: 20% in the Loser (Asset 0), 80% in the Winner (Asset 1)
    dynamic_strat = RebalancingStrategy('gradual', [0.5, 0.5], [0.2, 0.8], steps)
    
    # The 'static' mode ignores target weights and stays at 50/50
    static_strat = RebalancingStrategy('static', [0.5, 0.5], [0.2, 0.8], steps)

    # 4. Run them on the exact same rigged path
    val_dynamic = sim.run_fixed_path(fixed_prices, dynamic_strat)
    val_static = sim.run_fixed_path(fixed_prices, static_strat)

    # Calculate the exact dollar difference (The "Alpha")
    value_difference = val_dynamic - val_static

    # 5. Plot the showdown
    plt.figure(figsize=(10, 8))

    # Top Chart: The Macro View
    plt.subplot(2, 1, 1)
    plt.plot(val_dynamic, label="Dynamic (Gradual to 20/80)", color='green', linewidth=3)
    plt.plot(val_static, label="Static AMM (Stuck at 50/50)", color='gray', linestyle='--') 
    plt.title("Brownian Bridge: Dynamic Weights vs. Static AMM")
    plt.ylabel("Value ($)")
    plt.legend()

    # Bottom Chart: The "Alpha" View
    plt.subplot(2, 1, 2)
    plt.plot(value_difference, label="Dynamic Edge ($ Won)", color='blue', linewidth=2)
    plt.title("The 'Alpha' (Dynamic Value minus Static Value)")
    plt.xlabel("Time Step")
    plt.ylabel("Difference ($)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run it!
if __name__ == "__main__":
    EXPERIMENT_TO_RUN = "bridge"
    if EXPERIMENT_TO_RUN == "friday":
        print("Running Gradual vs Immediate")
        price_path_comparison()
    elif EXPERIMENT_TO_RUN == "bridge":
        print("Running Brownian Bridge: Dynamic vs Static")
        bridge_path_comparison()
    elif EXPERIMENT_TO_RUN == "monte_carlo":
        print("Running full Monte Carlo Simulation")
        runner = ExperimentRunner()
        runner.run_experiment(trials=1000, steps=200, portfolio_size=2)
        runner.print_summary_statistics()
        runner.plot_results()


    

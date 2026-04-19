"""
monte_carlo.py
--------------
European Option Pricing using Monte Carlo Simulation.

The core idea: simulate thousands of possible stock price paths using
Geometric Brownian Motion (GBM), compute the option payoff for each
path, and discount the average back to today.

    dS = S * (r*dt + sigma*dW)    where dW ~ N(0, dt)

Author : Paul Adeyemi
Project: Option Pricing Model — Black-Scholes & Monte Carlo
"""

import numpy as np


# ── Stock Path Simulation ─────────────────────────────────────────────────────

def simulate_gbm(S, T, r, sigma, n_simulations=100_000, n_steps=252, seed=42):
    """
    Simulate stock price paths using Geometric Brownian Motion.

    Parameters
    ----------
    S             : float — Current stock price
    T             : float — Time to expiration (in years)
    r             : float — Risk-free rate (annual, decimal)
    sigma         : float — Volatility (annual, decimal)
    n_simulations : int   — Number of Monte Carlo paths (default 100,000)
    n_steps       : int   — Number of time steps (default 252 trading days)
    seed          : int   — Random seed for reproducibility

    Returns
    -------
    np.ndarray : shape (n_steps+1, n_simulations) — all simulated paths
    """
    np.random.seed(seed)

    dt          = T / n_steps
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_steps, n_simulations)

    # Prepend zeros so the first row is the starting price
    log_returns = np.vstack([np.zeros(n_simulations), log_returns])
    price_paths = S * np.exp(np.cumsum(log_returns, axis=0))

    return price_paths


# ── Option Pricing via Monte Carlo ────────────────────────────────────────────

def monte_carlo_price(S, K, T, r, sigma, option_type="call",
                      n_simulations=100_000, n_steps=252, seed=42):
    """
    Price a European option using Monte Carlo simulation.

    Steps:
      1. Simulate n_simulations stock price paths via GBM
      2. Compute terminal payoff for each path
      3. Discount the average payoff to present value

    Parameters
    ----------
    S             : float — Current stock price
    K             : float — Strike price
    T             : float — Time to expiration (in years)
    r             : float — Risk-free rate (annual, decimal)
    sigma         : float — Volatility (annual, decimal)
    option_type   : str   — 'call' or 'put'
    n_simulations : int   — Number of simulated paths
    n_steps       : int   — Number of time steps per path
    seed          : int   — Random seed

    Returns
    -------
    dict :
        price       — Monte Carlo estimated option price
        std_error   — Standard error of the estimate
        conf_interval — 95% confidence interval (lower, upper)
    """
    paths           = simulate_gbm(S, T, r, sigma, n_simulations, n_steps, seed)
    terminal_prices = paths[-1]  # final stock prices at expiry

    if option_type.lower() == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Discount payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs

    price      = discounted_payoffs.mean()
    std_error  = discounted_payoffs.std() / np.sqrt(n_simulations)
    conf_lower = price - 1.96 * std_error
    conf_upper = price + 1.96 * std_error

    return {
        "price":         round(price,      4),
        "std_error":     round(std_error,  6),
        "conf_interval": (round(conf_lower, 4), round(conf_upper, 4)),
        "n_simulations": n_simulations,
    }


# ── Convergence Analysis ──────────────────────────────────────────────────────

def convergence_analysis(S, K, T, r, sigma, option_type="call",
                         sim_range=None, seed=42):
    """
    Analyse how the Monte Carlo price converges as n_simulations increases.
    Useful for visualising simulation accuracy vs computational cost.

    Parameters
    ----------
    sim_range : list of ints — simulation counts to test
                Default: [100, 500, 1000, 5000, 10000, 50000, 100000]

    Returns
    -------
    dict : {'n_simulations': [...], 'prices': [...], 'std_errors': [...]}
    """
    if sim_range is None:
        sim_range = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    prices     = []
    std_errors = []

    for n in sim_range:
        result = monte_carlo_price(S, K, T, r, sigma, option_type, n, seed=seed)
        prices.append(result["price"])
        std_errors.append(result["std_error"])

    return {
        "n_simulations": sim_range,
        "prices":        prices,
        "std_errors":    std_errors,
    }


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from black_scholes import black_scholes_price

    S, K, T, r, sigma = 150.0, 155.0, 1.0, 0.05, 0.20

    mc_call = monte_carlo_price(S, K, T, r, sigma, "call")
    mc_put  = monte_carlo_price(S, K, T, r, sigma, "put")
    bs_call = black_scholes_price(S, K, T, r, sigma, "call")
    bs_put  = black_scholes_price(S, K, T, r, sigma, "put")

    print("=" * 55)
    print("  MONTE CARLO vs BLACK-SCHOLES COMPARISON")
    print("=" * 55)
    print(f"  {'':25} {'CALL':>10}  {'PUT':>10}")
    print("-" * 55)
    print(f"  {'Black-Scholes Price':<25} ${bs_call:>9.4f}  ${bs_put:>9.4f}")
    print(f"  {'Monte Carlo Price':<25} ${mc_call['price']:>9.4f}  ${mc_put['price']:>9.4f}")
    print(f"  {'MC Std Error':<25}  {mc_call['std_error']:>9.6f}   {mc_put['std_error']:>9.6f}")
    print(f"  {'MC 95% CI':<25}  {str(mc_call['conf_interval']):>20}")
    print(f"  {'Simulations':<25}  {mc_call['n_simulations']:>9,}")
    print("=" * 55)

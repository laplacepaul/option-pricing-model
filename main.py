"""
main.py
-------
Run the full Option Pricing analysis — Black-Scholes + Monte Carlo.

Usage (Google Colab or local):
    python main.py

Or in Colab, run each section cell by cell after uploading the scripts.

Author : Paul Adeyemi
Project: Option Pricing Model — Black-Scholes & Monte Carlo
"""

from black_scholes  import black_scholes_price, greeks, put_call_parity_check
from monte_carlo    import monte_carlo_price, convergence_analysis
from visualizations import (
    plot_payoff_diagrams,
    plot_bs_price_vs_stock,
    plot_gbm_paths,
    plot_mc_convergence,
    plot_greeks,
)


# ── Parameters ────────────────────────────────────────────────────────────────
# Modify these to explore different scenarios

S     = 150.0   # Current stock price ($)
K     = 155.0   # Strike price ($)
T     = 1.0     # Time to expiration (1 year)
r     = 0.05    # Risk-free rate (5%)
sigma = 0.20    # Volatility (20%)


# ── 1. Black-Scholes Pricing ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  OPTION PRICING MODEL")
print("  Black-Scholes & Monte Carlo Simulation")
print("  Author: Paul Adeyemi")
print("=" * 60)

print(f"\n  Parameters:")
print(f"    Stock Price (S)   : ${S}")
print(f"    Strike Price (K)  : ${K}")
print(f"    Time to Expiry (T): {T} year(s)")
print(f"    Risk-Free Rate (r): {r*100:.1f}%")
print(f"    Volatility (σ)    : {sigma*100:.1f}%")

bs_call = black_scholes_price(S, K, T, r, sigma, "call")
bs_put  = black_scholes_price(S, K, T, r, sigma, "put")

print(f"\n{'─'*60}")
print(f"  BLACK-SCHOLES RESULTS")
print(f"{'─'*60}")
print(f"  Call Option Price : ${bs_call:.4f}")
print(f"  Put Option Price  : ${bs_put:.4f}")


# ── 2. Greeks ─────────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  GREEKS (Call Option)")
print(f"{'─'*60}")
call_greeks = greeks(S, K, T, r, sigma, "call")
greek_desc = {
    "delta": "Price change per $1 move in stock",
    "gamma": "Rate of change of Delta",
    "theta": "Price decay per day (time erosion)",
    "vega":  "Price change per 1% move in volatility",
    "rho":   "Price change per 1% move in interest rate",
}
for g, val in call_greeks.items():
    print(f"  {g.capitalize():<8}: {val:>10}   ({greek_desc[g]})")


# ── 3. Put-Call Parity ────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  PUT-CALL PARITY VERIFICATION")
print(f"{'─'*60}")
parity = put_call_parity_check(S, K, T, r, sigma)
for key, val in parity.items():
    print(f"  {key:<22}: {val}")


# ── 4. Monte Carlo Pricing ────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  MONTE CARLO SIMULATION (100,000 paths)")
print(f"{'─'*60}")
mc_call = monte_carlo_price(S, K, T, r, sigma, "call", n_simulations=100_000)
mc_put  = monte_carlo_price(S, K, T, r, sigma, "put",  n_simulations=100_000)

print(f"  {'Metric':<25} {'Call':>12}  {'Put':>12}")
print(f"  {'─'*50}")
print(f"  {'MC Price':<25} ${mc_call['price']:>11.4f}  ${mc_put['price']:>11.4f}")
print(f"  {'BS Price (reference)':<25} ${bs_call:>11.4f}  ${bs_put:>11.4f}")
print(f"  {'Difference':<25}  {abs(mc_call['price']-bs_call):>11.4f}   {abs(mc_put['price']-bs_put):>11.4f}")
print(f"  {'Std Error':<25}  {mc_call['std_error']:>11.6f}   {mc_put['std_error']:>11.6f}")
ci = mc_call['conf_interval']
print(f"  {'95% CI':<25}  ({float(ci[0]):.4f}, {float(ci[1]):.4f})")
print(f"  {'Simulations':<25}  {mc_call['n_simulations']:>11,}")


# ── 5. Visualizations ─────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  GENERATING CHARTS...")
print(f"{'─'*60}\n")

plot_payoff_diagrams(K=K)
plot_bs_price_vs_stock(K=K, T=T, r=r, sigma=sigma)
plot_gbm_paths(S=S, T=T, r=r, sigma=sigma)
plot_mc_convergence(S=S, K=K, T=T, r=r, sigma=sigma)
plot_greeks(K=K, T=T, r=r, sigma=sigma)

print("\n" + "=" * 60)
print("  Analysis complete. All charts saved.")
print("=" * 60)

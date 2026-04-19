"""
visualizations.py
-----------------
All charts and plots for the Option Pricing project.

Charts included:
  1. Option payoff diagrams (call & put)
  2. Black-Scholes price vs stock price (call & put)
  3. Sample GBM stock price paths
  4. Monte Carlo convergence plot
  5. Greeks sensitivity plots

Author : Paul Adeyemi
Project: Option Pricing Model — Black-Scholes & Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from black_scholes import black_scholes_price, greeks
from monte_carlo   import simulate_gbm, convergence_analysis


# ── Style ─────────────────────────────────────────────────────────────────────

BLUE   = "#185FA5"
RED    = "#C0392B"
GREEN  = "#1D9E75"
AMBER  = "#E67E22"
GRAY   = "#95A5A6"
DARK   = "#2C3E50"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.color":        "#ECF0F1",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


# ── 1. Payoff Diagrams ────────────────────────────────────────────────────────

def plot_payoff_diagrams(K=155.0, premium_call=10.0, premium_put=8.0):
    """
    Plot payoff at expiry for a long call and long put.
    Shows both gross payoff and net profit (after premium).
    """
    S_range = np.linspace(K * 0.5, K * 1.5, 300)

    call_payoff   = np.maximum(S_range - K, 0)
    put_payoff    = np.maximum(K - S_range, 0)
    call_profit   = call_payoff - premium_call
    put_profit    = put_payoff  - premium_put

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Option Payoff Diagrams at Expiry", fontsize=15, fontweight="bold", color=DARK, y=1.02)

    for ax, payoff, profit, color, label, premium in [
        (axes[0], call_payoff, call_profit, BLUE,  "Call Option", premium_call),
        (axes[1], put_payoff,  put_profit,  RED,   "Put Option",  premium_put),
    ]:
        ax.plot(S_range, payoff,  color=color,  linewidth=2,   label="Gross Payoff")
        ax.plot(S_range, profit,  color=color,  linewidth=2,   label=f"Net Profit (premium = ${premium})", linestyle="--")
        ax.axhline(0,  color=DARK,  linewidth=0.8, linestyle="-")
        ax.axvline(K,  color=GRAY,  linewidth=1,   linestyle=":", label=f"Strike K = ${K}")
        ax.fill_between(S_range, profit, 0, where=(profit > 0), alpha=0.12, color=GREEN, label="Profit zone")
        ax.fill_between(S_range, profit, 0, where=(profit < 0), alpha=0.12, color=RED,   label="Loss zone")
        ax.set_title(label, fontweight="bold", color=color)
        ax.set_xlabel("Stock Price at Expiry ($)")
        ax.set_ylabel("Payoff / Profit ($)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("payoff_diagrams.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: payoff_diagrams.png")


# ── 2. BS Price vs Stock Price ────────────────────────────────────────────────

def plot_bs_price_vs_stock(K=155.0, T=1.0, r=0.05, sigma=0.20):
    """
    Plot how the Black-Scholes option price changes as stock price varies.
    Shows both call and put prices alongside intrinsic value.
    """
    S_range    = np.linspace(80, 230, 300)
    call_prices = [black_scholes_price(s, K, T, r, sigma, "call") for s in S_range]
    put_prices  = [black_scholes_price(s, K, T, r, sigma, "put")  for s in S_range]
    intrinsic_call = np.maximum(S_range - K, 0)
    intrinsic_put  = np.maximum(K - S_range, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Black-Scholes Price vs Stock Price", fontsize=15, fontweight="bold", color=DARK, y=1.02)

    for ax, bs_prices, intrinsic, color, label in [
        (axes[0], call_prices, intrinsic_call, BLUE, "Call Option"),
        (axes[1], put_prices,  intrinsic_put,  RED,  "Put Option"),
    ]:
        ax.plot(S_range, bs_prices,  color=color, linewidth=2.5, label="BS Price (with time value)")
        ax.plot(S_range, intrinsic,  color=GRAY,  linewidth=1.5, linestyle="--", label="Intrinsic Value")
        ax.fill_between(S_range, bs_prices, intrinsic, alpha=0.1, color=color, label="Time Value")
        ax.axvline(K, color=GRAY, linewidth=1, linestyle=":", label=f"Strike K = ${K}")
        ax.set_title(label, fontweight="bold", color=color)
        ax.set_xlabel("Current Stock Price ($)")
        ax.set_ylabel("Option Price ($)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("bs_price_vs_stock.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: bs_price_vs_stock.png")


# ── 3. Simulated GBM Paths ───────────────────────────────────────────────────

def plot_gbm_paths(S=150.0, T=1.0, r=0.05, sigma=0.20,
                  n_paths=200, n_steps=252, seed=42):
    """
    Visualise a sample of simulated Geometric Brownian Motion stock price paths.
    """
    paths = simulate_gbm(S, T, r, sigma, n_simulations=n_paths,
                         n_steps=n_steps, seed=seed)
    time_axis = np.linspace(0, T, n_steps + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_axis, paths[:, :50], linewidth=0.5, alpha=0.4, color=BLUE)
    ax.plot(time_axis, paths.mean(axis=1), color=DARK, linewidth=2, label="Mean Path")
    ax.axhline(S, color=GRAY, linewidth=1, linestyle="--", label=f"Starting Price ${S}")
    ax.set_title(f"Simulated Stock Price Paths — Geometric Brownian Motion\n"
                 f"({n_paths} paths | σ={sigma*100:.0f}% | r={r*100:.0f}%)",
                 fontweight="bold", color=DARK)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Stock Price ($)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("gbm_paths.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: gbm_paths.png")


# ── 4. Monte Carlo Convergence ───────────────────────────────────────────────

def plot_mc_convergence(S=150.0, K=155.0, T=1.0, r=0.05, sigma=0.20):
    """
    Show how the Monte Carlo price converges toward the Black-Scholes
    analytical price as the number of simulations increases.
    """
    from black_scholes import black_scholes_price

    conv   = convergence_analysis(S, K, T, r, sigma, "call")
    bs_ref = black_scholes_price(S, K, T, r, sigma, "call")

    n_sims  = conv["n_simulations"]
    prices  = conv["prices"]
    errors  = conv["std_errors"]
    upper   = [p + 1.96 * e for p, e in zip(prices, errors)]
    lower   = [p - 1.96 * e for p, e in zip(prices, errors)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(n_sims, prices, "o-", color=BLUE,  linewidth=2, markersize=6, label="MC Price")
    ax.fill_between(n_sims, lower, upper, alpha=0.15, color=BLUE, label="95% Confidence Interval")
    ax.axhline(bs_ref, color=RED, linewidth=1.5, linestyle="--", label=f"Black-Scholes = ${bs_ref:.4f}")
    ax.set_title("Monte Carlo Convergence to Black-Scholes Price",
                 fontweight="bold", color=DARK)
    ax.set_xlabel("Number of Simulations (log scale)")
    ax.set_ylabel("Estimated Call Price ($)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: mc_convergence.png")


# ── 5. Greeks Sensitivity ────────────────────────────────────────────────────

def plot_greeks(K=155.0, T=1.0, r=0.05, sigma=0.20):
    """
    Plot Delta, Gamma, Theta, and Vega as functions of stock price.
    """
    S_range = np.linspace(80, 230, 300)

    delta_call = [greeks(s, K, T, r, sigma, "call")["delta"] for s in S_range]
    delta_put  = [greeks(s, K, T, r, sigma, "put") ["delta"] for s in S_range]
    gamma      = [greeks(s, K, T, r, sigma, "call")["gamma"] for s in S_range]
    theta_call = [greeks(s, K, T, r, sigma, "call")["theta"] for s in S_range]
    vega       = [greeks(s, K, T, r, sigma, "call")["vega"]  for s in S_range]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Option Greeks — Sensitivity Analysis", fontsize=15,
                 fontweight="bold", color=DARK)

    plots = [
        (axes[0,0], delta_call, BLUE,  "Delta (Call)",  "Delta"),
        (axes[0,1], gamma,      GREEN, "Gamma",         "Gamma"),
        (axes[1,0], theta_call, AMBER, "Theta (per day)","Theta ($)"),
        (axes[1,1], vega,       RED,   "Vega (per 1% vol)","Vega ($)"),
    ]

    for ax, values, color, title, ylabel in plots:
        ax.plot(S_range, values, color=color, linewidth=2)
        ax.axvline(K, color=GRAY, linewidth=1, linestyle=":", label=f"Strike K=${K}")
        ax.set_title(title, fontweight="bold", color=color)
        ax.set_xlabel("Stock Price ($)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    # Overlay put delta on first plot
    axes[0,0].plot(S_range, delta_put, color=RED, linewidth=2,
                   linestyle="--", label="Delta (Put)")
    axes[0,0].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("greeks_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: greeks_sensitivity.png")


# ── Run All ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all charts...\n")
    plot_payoff_diagrams()
    plot_bs_price_vs_stock()
    plot_gbm_paths()
    plot_mc_convergence()
    plot_greeks()
    print("\nAll charts saved successfully.")

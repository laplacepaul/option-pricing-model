"""
black_scholes.py
----------------
European Option Pricing using the Black-Scholes Model.

Author : Paul Adeyemi
Project: Option Pricing Model — Black-Scholes & Monte Carlo
"""

import numpy as np
from scipy.stats import norm


# ── Core Formula ─────────────────────────────────────────────────────────────

def d1(S, K, T, r, sigma):
    """
    Compute d1 component of the Black-Scholes formula.

    Parameters
    ----------
    S     : float — Current stock price
    K     : float — Strike price
    T     : float — Time to expiration (in years)
    r     : float — Risk-free interest rate (annual, decimal)
    sigma : float — Volatility of the underlying asset (annual, decimal)

    Returns
    -------
    float
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """
    Compute d2 component of the Black-Scholes formula.
    d2 = d1 - sigma * sqrt(T)
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price for a European option.

    Parameters
    ----------
    S           : float — Current stock price
    K           : float — Strike price
    T           : float — Time to expiration (in years)
    r           : float — Risk-free interest rate (annual, decimal)
    sigma       : float — Volatility (annual, decimal)
    option_type : str   — 'call' or 'put'

    Returns
    -------
    float : Option price
    """
    if T <= 0:
        raise ValueError("Time to expiration T must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be greater than 0.")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price S and strike price K must be positive.")

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        price = S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return price


# ── Greeks ───────────────────────────────────────────────────────────────────

def greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes Greeks for a European option.

    Greeks measure the sensitivity of the option price to various parameters:
      Delta : sensitivity to stock price changes
      Gamma : rate of change of Delta
      Theta : sensitivity to time decay (per day)
      Vega  : sensitivity to volatility changes (per 1% move)
      Rho   : sensitivity to interest rate changes (per 1% move)

    Returns
    -------
    dict : {'delta', 'gamma', 'theta', 'vega', 'rho'}
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    gamma = norm.pdf(_d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(_d1) * np.sqrt(T) / 100

    if option_type.lower() == "call":
        delta = norm.cdf(_d1)
        theta = (-(S * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(_d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(_d2) / 100

    elif option_type.lower() == "put":
        delta = norm.cdf(_d1) - 1
        theta = (-(S * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-_d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-_d2) / 100

    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega":  round(vega,  6),
        "rho":   round(rho,   6),
    }


# ── Put-Call Parity Check ─────────────────────────────────────────────────────

def put_call_parity_check(S, K, T, r, sigma):
    """
    Verify Put-Call Parity:
        C - P = S - K * exp(-rT)

    Returns the difference — should be near zero for valid pricing.
    """
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put  = black_scholes_price(S, K, T, r, sigma, "put")
    lhs  = call - put
    rhs  = S - K * np.exp(-r * T)
    return {
        "call_price":      round(call, 4),
        "put_price":       round(put,  4),
        "LHS (C - P)":     round(lhs,  6),
        "RHS (S - Ke-rT)": round(rhs,  6),
        "difference":      round(abs(lhs - rhs), 10),
        "parity_holds":    np.isclose(lhs, rhs, atol=1e-6),
    }


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    S, K, T, r, sigma = 150.0, 155.0, 1.0, 0.05, 0.20

    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    put_price  = black_scholes_price(S, K, T, r, sigma, "put")
    g          = greeks(S, K, T, r, sigma, "call")
    parity     = put_call_parity_check(S, K, T, r, sigma)

    print("=" * 50)
    print("  BLACK-SCHOLES OPTION PRICING")
    print("=" * 50)
    print(f"  Stock Price (S)  : ${S}")
    print(f"  Strike Price (K) : ${K}")
    print(f"  Time to Expiry   : {T} year(s)")
    print(f"  Risk-Free Rate   : {r*100}%")
    print(f"  Volatility       : {sigma*100}%")
    print("-" * 50)
    print(f"  Call Price       : ${call_price:.4f}")
    print(f"  Put Price        : ${put_price:.4f}")
    print("-" * 50)
    print("  GREEKS (Call):")
    for k, v in g.items():
        print(f"    {k.capitalize():<8}: {v}")
    print("-" * 50)
    print("  PUT-CALL PARITY CHECK:")
    for k, v in parity.items():
        print(f"    {k:<20}: {v}")
    print("=" * 50)

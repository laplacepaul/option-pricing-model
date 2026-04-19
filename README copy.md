# Option Pricing Model — Black-Scholes & Monte Carlo

**Author:** Paul Adeyemi  
**Stack:** Python · NumPy · SciPy · Matplotlib  
**Topics:** Quantitative Finance · Derivatives Pricing · Stochastic Processes · Financial Mathematics

---

## Overview

Implementation of two classical approaches to pricing European options:

- **Black-Scholes Model** — closed-form analytical solution derived from the Black-Scholes-Merton PDE
- **Monte Carlo Simulation** — numerical pricing via Geometric Brownian Motion across 100,000 simulated paths

Both methods are benchmarked against each other, with Put-Call Parity verified analytically and convergence of the MC estimate to the BS price demonstrated empirically.

---

## Mathematical Foundation

### Black-Scholes Formula

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)      (Call)
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)    (Put)

d₁ = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d₂ = d₁ - σ·√T
```

| Symbol | Description |
|--------|-------------|
| S | Current stock price |
| K | Strike price |
| T | Time to expiration (years) |
| r | Risk-free interest rate |
| σ | Volatility of the underlying |
| N(·) | Cumulative standard normal distribution |

### Geometric Brownian Motion (Monte Carlo)

Stock price dynamics modelled as:

```
dS = S·(r·dt + σ·dW)     where dW ~ N(0, dt)
```

Discrete simulation:

```
S(t+dt) = S(t) · exp[ (r - σ²/2)·dt + σ·√dt·Z ]     Z ~ N(0,1)
```

Option price estimated as the discounted expected payoff:

```
C ≈ e^(-rT) · E[ max(S_T - K, 0) ]
```

---

## Project Structure

```
option-pricing-model/
│
├── black_scholes.py     # BS formula, Greeks, Put-Call Parity
├── monte_carlo.py       # GBM simulation, MC pricing, convergence analysis
├── visualizations.py    # Charts and plots
├── main.py              # Full analysis runner
└── README.md
```

---

## Results

Parameters used: `S=150, K=155, T=1yr, r=5%, σ=20%`

| Method | Call Price | Put Price |
|--------|-----------|-----------|
| Black-Scholes | $13.1698 | $10.6104 |
| Monte Carlo (100k paths) | $13.1369 | $10.6138 |
| Difference | $0.0329 | $0.0034 |

Put-Call Parity verified: `C - P = S - Ke^(-rT)` → difference = **0.0** ✅

---

## Charts

| Chart | Description |
|-------|-------------|
| `payoff_diagrams.png` | Call & put payoff and net profit at expiry |
| `bs_price_vs_stock.png` | BS price vs stock price with intrinsic value |
| `gbm_paths.png` | 200 simulated GBM stock price paths |
| `mc_convergence.png` | MC price convergence toward BS analytical solution |
| `greeks_sensitivity.png` | Delta, Gamma, Theta, Vega vs stock price |

---

## Key Concepts

- **Stochastic calculus** — Ito's lemma underpins the Black-Scholes PDE derivation
- **Risk-neutral pricing** — both models price under the risk-neutral measure
- **Law of large numbers** — MC price converges to BS as simulations increase
- **Put-Call Parity** — `C - P = S - Ke^(-rT)`
- **Greeks** — first and second order sensitivities of option price to market variables

---

## Dependencies

```bash
pip install numpy scipy matplotlib
```

---

## License

MIT License

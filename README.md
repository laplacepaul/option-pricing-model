# Option Pricing Model — Black-Scholes & Monte Carlo Simulation

**Author:** Paul Adeyemi  
**Stack:** Python · NumPy · SciPy · Matplotlib  
**Topics:** Quantitative Finance · Stochastic Processes · Derivatives Pricing · Financial Mathematics

---

## Overview

This project implements two classical approaches to pricing European options:

1. **Black-Scholes Model** — analytical closed-form solution derived from the Black-Scholes-Merton PDE
2. **Monte Carlo Simulation** — numerical approach using Geometric Brownian Motion to simulate thousands of possible stock price paths

Both methods are compared side by side, and the mathematical relationship between them is explored through convergence analysis and Put-Call Parity verification.

---

## Mathematical Foundation

### Black-Scholes Formula

The price of a European **call** option is given by:

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
```

And a European **put** option:

```
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

Where:
```
d₁ = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d₂ = d₁ - σ·√T
```

| Symbol | Description |
|--------|-------------|
| S      | Current stock price |
| K      | Strike price |
| T      | Time to expiration (years) |
| r      | Risk-free interest rate |
| σ      | Volatility of the underlying asset |
| N(·)   | Cumulative standard normal distribution |

### Geometric Brownian Motion (Monte Carlo)

Stock price dynamics are modelled using GBM:

```
dS = S·(r·dt + σ·dW)    where dW ~ N(0, dt)
```

The discrete approximation used in simulation:

```
S(t+dt) = S(t) · exp[ (r - σ²/2)·dt + σ·√dt·Z ]    Z ~ N(0,1)
```

The option price is then estimated as the **discounted expected payoff**:

```
C ≈ e^(-rT) · E[max(S_T - K, 0)]
```

---

## Project Structure

```
option-pricing-model/
│
├── black_scholes.py       # BS formula, Greeks, Put-Call Parity
├── monte_carlo.py         # GBM simulation, MC pricing, convergence analysis
├── visualizations.py      # All charts and plots
├── main.py                # Runs the full analysis
└── README.md
```

---

## Features

- European call and put pricing via Black-Scholes
- Option Greeks (Delta, Gamma, Theta, Vega, Rho)
- Put-Call Parity verification
- Monte Carlo simulation with 100,000 paths
- 95% confidence intervals on MC estimates
- Convergence analysis (MC vs BS analytical solution)
- 5 publication-quality charts

---

## Charts Generated

| Chart | Description |
|-------|-------------|
| `payoff_diagrams.png`    | Call & put payoff/profit at expiry |
| `bs_price_vs_stock.png`  | BS option price across stock price range |
| `gbm_paths.png`          | 200 simulated GBM stock price paths |
| `mc_convergence.png`     | MC price convergence toward BS solution |
| `greeks_sensitivity.png` | Delta, Gamma, Theta, Vega vs stock price |

---

## Quickstart (Google Colab)

```python
# Step 1 — Upload all .py files to Colab or clone the repo
# Step 2 — Install dependencies (already available in Colab)
!pip install numpy scipy matplotlib

# Step 3 — Run the full analysis
!python main.py
```

Or run interactively — copy each module into a Colab cell and call functions directly:

```python
from black_scholes import black_scholes_price, greeks

price = black_scholes_price(S=150, K=155, T=1.0, r=0.05, sigma=0.20, option_type="call")
print(f"Call Price: ${price:.4f}")
```

---

## Example Output

```
============================================================
  OPTION PRICING MODEL
  Black-Scholes & Monte Carlo Simulation
  Author: Paul Adeyemi
============================================================

  Parameters:
    Stock Price (S)   : $150.0
    Strike Price (K)  : $155.0
    Time to Expiry (T): 1.0 year(s)
    Risk-Free Rate (r): 5.0%
    Volatility (σ)    : 20.0%

────────────────────────────────────────────────────────────
  BLACK-SCHOLES RESULTS
────────────────────────────────────────────────────────────
  Call Option Price : $12.1673
  Put  Option Price : $ 9.6285

────────────────────────────────────────────────────────────
  MONTE CARLO SIMULATION (100,000 paths)
────────────────────────────────────────────────────────────
  Metric                          Call           Put
  ──────────────────────────────────────────────────
  MC Price                     $12.1521       $9.6089
  BS Price (reference)         $12.1673       $9.6285
  Difference                     0.0152         0.0196
  Std Error                    0.000381       0.000302
  95% CI                (12.1515, 12.1527)
  Simulations                   100,000
```

---

## Key Concepts Demonstrated

- **Stochastic calculus** — Ito's lemma underpins the Black-Scholes PDE derivation
- **Risk-neutral pricing** — both models price under the risk-neutral measure
- **Law of large numbers** — MC price converges to BS as simulations increase
- **Put-Call Parity** — verified analytically: `C - P = S - K·e^(-rT)`
- **Greeks** — first and second order sensitivities of option price

---

## Dependencies

```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

---

## License

MIT License — free to use, modify, and distribute.

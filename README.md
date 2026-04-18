# GA-Based Dynamic Bandwidth Allocation

Optimises bandwidth allocation across N users with heterogeneous demands and priorities using a Genetic Algorithm. Models dynamic network conditions with a sinusoidal bandwidth model and evaluates fairness using Jain's Fairness Index.

---

## Problem Statement

Given N users each with a bandwidth demand and a QoS priority level (1–5), allocate a limited and time-varying network bandwidth to **maximise priority-weighted satisfaction** while respecting hard capacity constraints and minimum QoS guarantees.

This is a constrained continuous optimisation problem — the GA is used because the objective is non-linear when extended with fairness terms, and the dynamic constraint makes closed-form solutions impractical.

---

## Key Features

**Dynamic Bandwidth Constraint**  
Available bandwidth is not fixed. It follows a sinusoidal model:

```
B_t = B_base + 0.15 * B_base * sin(2π * gen / 50)
```

This simulates real network congestion cycles (±15% swing, period = 50 generations). The GA adapts its repair and feasibility checks to the current B_t every generation.

**Jain's Fairness Index Tracking**  
Fairness is computed every generation as:

```
JFI = (Σxᵢ)² / (n · Σxᵢ²),   xᵢ = allocated_i / demand_i
```

JFI ∈ (0, 1]. Value of 1 = all users equally satisfied. Tracked per generation to show how fairness evolves during optimisation, not just as a final metric.

**Pareto Demand Distribution**  
User demands are sampled from a Pareto distribution (shape = 1.2) to model real internet traffic. Heavy-tail behaviour — a small number of users have disproportionately high demand — is well established in networking literature (Leland et al., 1994).

---

## GA Design

| Component | Choice | Reason |
|---|---|---|
| Encoding | Real-valued vector | Continuous allocation variables |
| Fitness | Weighted satisfaction ratio | Rewards high-priority fulfilment |
| Selection | Tournament (k=3) | Balances exploration vs exploitation |
| Crossover | Arithmetic (convex combination) | Offspring stay within parent range |
| Mutation | Gaussian perturbation (σ=1.0) | Smooth local search |
| Constraint handling | Repair operator | Always maintains feasibility |
| Elitism | Top 5% preserved | Prevents loss of best solutions |
| Early stopping | No improvement for 20 gens | Avoids wasted compute |

**Repair Operator** (applied after every crossover and mutation):
1. Floor each allocation to 10% of that user's demand (QoS guarantee)
2. Clip to demand ceiling (no over-allocation per user)
3. If total exceeds B_t, scale down proportionally: `c = c * (B_t / Σc)`

---

## Output

**Console summary:**
```
Best Fitness     : 12.3110
Total Allocated  : 101.88 / 101.88 Mbps  (100.0% utilisation)
Avg Satisfaction : 82.2%
Jain's Fairness  : 0.9514  (1.0 = perfect)
Worst-served User: U5  (46.0% satisfied)

User    Priority      Demand        Allocated     Satisfaction
--------------------------------------------------------------
  U1    5             9.56 Mbps     8.80 Mbps     92.0%
  U2    1             60.00 Mbps    54.08 Mbps    90.1%
  U3    5             12.46 Mbps    10.56 Mbps    84.7%
  U4    1             12.67 Mbps    12.41 Mbps    97.9%
  U5    4             34.87 Mbps    16.04 Mbps    46.0%
```

**Plots generated:**
- `convergence_dynamic_bw.png` — fitness convergence overlaid with the dynamic bandwidth signal (dual axis)
- `jfi_over_generations.png` — Jain's Fairness Index evolution across generations
- `demand_vs_allocated.png` — per-user demand vs allocated bandwidth (grouped bar chart)

---

## Project Structure

```
ga-bandwidth-allocation/
│
├── src/
│   └── ga_bandwidth.py
│
├── outputs/
│   ├── convergence_dynamic_bw.png
│   ├── jfi_over_generations.png
│   └── demand_vs_allocated.png
│
└── README.md
```

---

## Setup and Usage

```bash
pip install numpy matplotlib
python src/ga_bandwidth.py
```

No other dependencies required.

**Inputs prompted at runtime:**
```
Enter number of users: 10
Enter base bandwidth (Mbps) [actual BW will vary ±15% dynamically]: 100
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `POP` | 50 | Population size |
| `GENS` | 200 | Max generations |
| `CR` | 0.8 | Crossover rate |
| `MR` | 0.1 | Mutation rate per gene |
| `SIGMA` | 1.0 | Gaussian mutation std dev |
| `K` | 3 | Tournament size |
| `ELITISM` | 0.05 | Elite fraction preserved |
| `NO_IMP` | 20 | Early stop patience |
| `B_AMP` | 15% of B | Dynamic BW swing amplitude |
| `B_PERIOD` | 50 | Dynamic BW cycle length (gens) |
| `PARETO_SHAPE` | 1.2 | Demand distribution shape |
| `SEED` | 42 | Random seed |

---

## References

- Jain, R. et al. (1984). *A quantitative measure of fairness and discrimination for resource allocation in shared computer systems*.
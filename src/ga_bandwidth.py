import numpy as np
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 48
np.random.seed(SEED)

# ── Input ────────────────────────────────────────────────────────────────────
N = int(input("Enter number of users: "))
B = float(input("Enter base bandwidth (Mbps) [actual BW will vary ±15% dynamically]: "))

# ── User data  (Pareto heavy-tail demand distribution) ───────────────────────
# Pareto distribution models real internet traffic (Leland et al., 1994)
# Shape parameter m=1.2 gives realistic heavy-tail behaviour
PARETO_SHAPE = 1.2
raw_demands  = (np.random.pareto(PARETO_SHAPE, N) + 1)   # shift so min = 1
demands      = raw_demands / raw_demands.max() * (B * 0.6) # scale to 60% of B
demands      = np.clip(demands, 0.5, B * 0.6)

priorities   = np.random.randint(1, 6, N)
weights      = 6 - priorities                             # higher priority → higher weight

print(f"\n{'User':<8}{'Demand (Mbps)':<18}{'Priority'}")
print("-" * 35)
for i in range(N):
    print(f"  U{i+1:<7}{demands[i]:<18.2f}{priorities[i]}")
print(f"\nBase Bandwidth : {B} Mbps")
print(f"Dynamic Range  : {B - B*0.15:.1f} – {B + B*0.15:.1f} Mbps  (±15% sinusoidal)")

# ── GA Parameters ────────────────────────────────────────────────────────────
POP     = 50
GENS    = 200
CR      = 0.8    # crossover rate
MR      = 0.1    # mutation rate
SIGMA   = 1.0    # Gaussian mutation std dev
K       = 3      # tournament size
ELITISM = 0.05   # elite fraction
NO_IMP  = 20     # early-stop patience

# ── Dynamic bandwidth parameters ─────────────────────────────────────────────
# B fluctuates sinusoidally to simulate real network congestion cycles
B_AMP    = B * 0.15          # ±15% swing around base
B_PERIOD = 50                # one full cycle every 50 generations

def dynamic_bandwidth(gen):
    """Return available bandwidth at generation `gen` using sinusoidal model."""
    return B + B_AMP * np.sin(2 * np.pi * gen / B_PERIOD)

# ── Jain's Fairness Index ────────────────────────────────────────────────────
def jains_fairness(alloc):
    """
    Compute Jain's Fairness Index on satisfaction ratios (alloc / demand).
    JFI = (Σxᵢ)² / (n · Σxᵢ²),  range (0, 1].  1 = perfect fairness.
    """
    x   = alloc / demands
    return (np.sum(x) ** 2) / (N * np.sum(x ** 2))

# ── Repair operator ──────────────────────────────────────────────────────────
def repair(c, B_t):
    """Project chromosome into feasible space given current bandwidth B_t."""
    c = np.maximum(c, 0.1 * demands)   # QoS floor: at least 10% of demand
    c = np.clip(c, 0, demands)         # never exceed demand
    t = np.sum(c)
    return c * (B_t / t) if t > B_t else c

# ── Fitness ──────────────────────────────────────────────────────────────────
def fitness(c):
    """Weighted satisfaction ratio: maximise priority-adjusted demand fulfilment."""
    return np.sum(weights * (c / demands))

# ── Tournament selection ─────────────────────────────────────────────────────
def select(pop, fits):
    """Return a copy of the fittest individual from K random candidates."""
    cands = np.random.choice(len(pop), K, replace=False)
    return pop[max(cands, key=lambda i: fits[i])].copy()

# ── Arithmetic crossover ─────────────────────────────────────────────────────
def crossover(p1, p2, B_t):
    """Produce two offspring via convex combination; repair to feasibility."""
    a = np.random.random()
    return repair(a*p1 + (1-a)*p2, B_t), repair(a*p2 + (1-a)*p1, B_t)

# ── Gaussian mutation ────────────────────────────────────────────────────────
def mutate(c, B_t):
    """Perturb each gene with probability MR; repair afterwards."""
    c = c.copy()
    mask     = np.random.random(N) < MR
    c[mask] += np.random.normal(0, SIGMA, mask.sum())
    return repair(c, B_t)

# ── Initialise population ────────────────────────────────────────────────────
B_init     = dynamic_bandwidth(0)
population = [repair(np.random.uniform(0, demands), B_init) for _ in range(POP)]

# ── GA loop ───────────────────────────────────────────────────────────────────
best_fits   = []
jfi_history = []
bw_history  = []

best_chrom  = None
best_fit    = -np.inf
best_B_t    = B       # bandwidth at the generation best_chrom was found
no_imp      = 0

for gen in range(GENS):
    B_t   = dynamic_bandwidth(gen)          # bandwidth available this generation
    fits  = [fitness(c) for c in population]
    idx   = int(np.argmax(fits))
    gf    = fits[idx]

    best_fits.append(gf)
    jfi_history.append(jains_fairness(population[idx]))
    bw_history.append(B_t)

    if gf > best_fit + 1e-4:
        best_fit, best_chrom, best_B_t, no_imp = gf, population[idx].copy(), B_t, 0
    else:
        no_imp += 1
    if no_imp >= NO_IMP:
        print(f"\nEarly stop at generation {gen}  (no improvement for {NO_IMP} gens)")
        break

    ec         = max(1, int(ELITISM * POP))
    sorted_pop = [c for _, c in sorted(zip(fits, population), key=lambda x: x[0], reverse=True)]
    new_pop    = [c.copy() for c in sorted_pop[:ec]]

    while len(new_pop) < POP:
        p1, p2 = select(population, fits), select(population, fits)
        if np.random.random() < CR:
            c1, c2 = crossover(p1, p2, B_t)
        else:
            c1, c2 = p1.copy(), p2.copy()
        new_pop.append(mutate(c1, B_t))
        if len(new_pop) < POP:
            new_pop.append(mutate(c2, B_t))

    population = new_pop

# ── Output ────────────────────────────────────────────────────────────────────
best_chrom    = repair(best_chrom, best_B_t)   # guarantee feasibility after float drift
total_alloc   = np.sum(best_chrom)
avg_sat       = np.mean(best_chrom / demands) * 100
min_sat_user  = int(np.argmin(best_chrom / demands))
final_jfi     = jains_fairness(best_chrom)

print(f"\n{'='*60}")
print(f"Best Fitness     : {best_fit:.4f}")
print(f"Total Allocated  : {total_alloc:.2f} / {best_B_t:.2f} Mbps  ({total_alloc/best_B_t*100:.1f}% utilisation)")
print(f"Avg Satisfaction : {avg_sat:.1f}%")
print(f"Jain's Fairness  : {final_jfi:.4f}  (1.0 = perfect)")
print(f"Worst-served User: U{min_sat_user+1}  ({best_chrom[min_sat_user]/demands[min_sat_user]*100:.1f}% satisfied)")
print(f"\n{'User':<8}{'Priority':<12}{'Demand':<14}{'Allocated':<14}{'Satisfaction'}")
print("-" * 60)
for i in range(N):
    print(f"  U{i+1:<7}{priorities[i]:<12}{demands[i]:<14.2f}{best_chrom[i]:<14.2f}{best_chrom[i]/demands[i]*100:.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
users = [f"U{i+1}" for i in range(N)]
gens  = range(len(best_fits))

# 1. Convergence + Dynamic BW (dual axis)
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(gens, best_fits,   color='steelblue', linewidth=2, label='Best Fitness')
ax2.plot(gens, bw_history,  color='orange',    linewidth=1.2, linestyle='--', alpha=0.7, label='Available BW')
ax1.set_xlabel("Generation")
ax1.set_ylabel("Best Fitness",         color='steelblue')
ax2.set_ylabel("Available BW (Mbps)",  color='orange')
ax1.set_title("Fitness Convergence under Dynamic Bandwidth", fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("convergence_dynamic_bw.png", dpi=150)
plt.show()

# 2. Jain's Fairness Index over generations
plt.figure(figsize=(9, 4))
plt.plot(gens, jfi_history, color='seagreen', linewidth=2)
plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Perfect Fairness')
plt.xlabel("Generation")
plt.ylabel("Jain's Fairness Index")
plt.title("Fairness Evolution over Generations", fontweight='bold')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("jfi_over_generations.png", dpi=150)
plt.show()

# 3. Demand vs Allocated per user
x, w = np.arange(N), 0.35
fig, ax = plt.subplots(figsize=(max(8, N), 5))
ax.bar(x - w/2, demands,    w, label='Demand',    color='steelblue', alpha=0.8)
ax.bar(x + w/2, best_chrom, w, label='Allocated', color='tomato',    alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(users)
ax.set_xlabel("Users")
ax.set_ylabel("Bandwidth (Mbps)")
ax.set_title("Demand vs Allocated Bandwidth per User", fontweight='bold')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("demand_vs_allocated.png", dpi=150)
plt.show()
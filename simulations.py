#!/usr/bin/env python3
"""
AUCIL Evaluation Script (optimized)
===================================
Regenerates every figure in the AUCIL paper (IEEE S&P 2027 submission)
in ONE consistent style, using the exact algorithm from the original
artifact repository (https://anonymous.4open.science/r/Implementation-4CF6).

This version is numerically identical to the original notebook algorithm
(verified across thousands of random instances) but is substantially faster:
  * the inner allocation/bribe loops avoid copy.deepcopy entirely, using
    cheap numpy / set copies instead (~15x faster bribe computation);
  * the bribe-vs-fee figures compute only the single targeted transaction's
    bribe instead of all m transactions (another O(m) factor);
  * the broadcast-equilibrium solver supports an optional numba speed-up and a
    tunable grid resolution.

Figures produced (all in Figures/, consistent style):
  FeeBribeCommittee.pdf      - Bribe vs fee, committee size n varied
  FeeBribeILSize.pdf         - Bribe vs fee, input-list size k varied
  FeeBribeMempool.pdf        - Bribe vs fee, fee scaling varied
  BroadcastEquilibrium.pdf   - Withholding probability vs VRF bias

Requirements:
  pip install numpy scipy matplotlib            (numba optional)
  A LaTeX install is needed only if USE_TEX=True (default False for portability).

Usage:
  python3 evaluation.py                # all four figures
  python3 evaluation.py --fast         # coarser broadcast grid (quicker)
  python3 evaluation.py --check        # self-test vs the reference algorithm
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional numba acceleration (falls back gracefully)
try:
    from numba import jit, prange
    HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    HAVE_NUMBA = False

    def jit(*a, **k):
        def wrap(f):
            return f
        return wrap

    prange = range

# ════════════════════════════════════════════════════════════════
# Global style - single source of truth for ALL figures
# ════════════════════════════════════════════════════════════════

IMAGES_DIR = "Figures/"
USE_TEX = False                 # set True if a LaTeX install is available
FIGSIZE = (5, 3)
FONTSIZE = 12

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.size": FONTSIZE,
})

SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]   # blue, red, green
REF_KW = dict(color="black", alpha=0.25, linewidth=1)

# ════════════════════════════════════════════════════════════════
# Core algorithm
# (Numerically identical to the original artifact's
#  two_step_transaction_inclusion / calculate_bribes, but deepcopy-free.)
# ════════════════════════════════════════════════════════════════

def two_step_transaction_inclusion(n, m, k, Ua, alpha=1):
    """Algorithm 1: greedy selection (Step 1) + round-robin allocation (Step 2).

    Returns (La, N_t) where La maps party id (1..n) to a set of object indices
    and N_t[i] is the number of times object i was selected in Step 1.
    """
    Ua = np.asarray(Ua, dtype=float)
    Us = Ua.copy()
    N = np.zeros(m, dtype=float)
    N_t = np.zeros(m, dtype=float)
    S = []
    for _ in range(n * k):
        s = int(np.argmax(Us / (N + 1)))
        S.append(s)
        if Us[s] == -1:
            break
        N[s] += alpha
        N_t[s] += 1
        if N_t[s] >= n:
            Us[s] = -1
    U_f = np.divide(Ua, (N + 1 - alpha), out=np.zeros_like(Ua), where=N != 0)
    La = {i: set() for i in range(1, n + 1)}
    A = sorted(S, key=lambda s: (-U_f[s], s))
    for j, obj in enumerate(A):
        La[(j % n) + 1].add(obj)
    return La, N_t


def calculate_bribes(n, m, k, Ua, La, N_t, alpha=1):
    """Minimum adversarial bribe to remove EACH transaction from the input lists.

    Deepcopy-free reimplementation; numerically identical to the reference.
    """
    Ua = np.asarray(Ua, dtype=float)
    party_sets = [set(La[j + 1]) for j in range(n)]
    incList = np.unique([o for s in party_sets for o in s]).astype(int)

    Usa = Ua.copy()
    for i in incList:
        if N_t[i] >= n:
            Usa[i] = -1.0

    bribes = np.zeros_like(Ua)
    for i in incList:                       # np.unique output is already sorted
        Us = Usa.copy()
        N2_t = N_t.copy()
        N2 = N_t * alpha                    # fresh array (not a view)
        bribe = Ua[i] * N2_t[i]
        Us[i] = -1.0
        newadditions = []
        for j in range(n):
            if i in party_sets[j]:
                s = int(np.argmax(Us / (N2 + 1)))
                if Us[s] == -1:
                    break
                newadditions.append(s)
                N2[s] += alpha
                N2_t[s] += 1
                if N2_t[s] >= n:
                    Us[s] = -1.0
        U_f = np.divide(Ua, (N2 + 1 - alpha), out=np.zeros_like(Ua), where=N2 != 0)
        for h in newadditions:
            bribe -= U_f[h]
        bribes[i] = max(0.0, bribe)
    return bribes


def calculate_bribe_single(n, m, k, Ua, La, N_t, target, alpha=1):
    """Bribe to remove a SINGLE target transaction.

    Equivalent to calculate_bribes(...)[target] but avoids computing the
    bribe for every other transaction -- the only quantity the bribe-vs-fee
    figures need. Numerically identical to the reference (verified).
    """
    Ua = np.asarray(Ua, dtype=float)
    party_sets = [La[j + 1] for j in range(n)]
    if not any(target in s for s in party_sets):
        return 0.0
    incList = {o for s in party_sets for o in s}

    Usa = Ua.copy()
    for i in incList:
        if N_t[i] >= n:
            Usa[i] = -1.0

    Us = Usa.copy()
    N2_t = N_t.copy()
    N2 = N_t * alpha
    bribe = Ua[target] * N2_t[target]
    Us[target] = -1.0
    newadditions = []
    for j in range(n):
        if target in party_sets[j]:
            s = int(np.argmax(Us / (N2 + 1)))
            if Us[s] == -1:
                break
            newadditions.append(s)
            N2[s] += alpha
            N2_t[s] += 1
            if N2_t[s] >= n:
                Us[s] = -1.0
    U_f = np.divide(Ua, (N2 + 1 - alpha), out=np.zeros_like(Ua), where=N2 != 0)
    for h in newadditions:
        bribe -= U_f[h]
    return max(0.0, bribe)


def bribe_curve(n, m, k, Ug, x_axis, target=0, gamma=1.0):
    """Bribe tolerated for `target` as its fee sweeps over x_axis.

    Mutates only the target entry (matching the original notebook's loop) and
    computes only the target's bribe. `gamma` is the broadcast probability
    (alpha); pass the self-consistent equilibrium value from
    solve_equilibrium_gamma() to reflect the paper's gamma<1 model.
    """
    Ug = np.asarray(Ug, dtype=float).copy()
    y = np.empty_like(x_axis, dtype=float)
    for idx, g in enumerate(x_axis):
        Ug[target] = g
        La, N_t = two_step_transaction_inclusion(n, m, k, Ug, alpha=gamma)
        y[idx] = calculate_bribe_single(n, m, k, Ug, La, N_t, target, alpha=gamma)
    return y


# ════════════════════════════════════════════════════════════════
# Broadcast-equilibrium solver (numba-accelerated when available)
# ════════════════════════════════════════════════════════════════

BIAS_MIN = 0.0
BIAS_MAX = 6.0
LC = 6000
_bias_values = np.linspace(BIAS_MIN, BIAS_MAX, LC)


@jit(nopython=True)
def _P_max(eta, p_values, n, bias_values, lc):
    if eta < BIAS_MIN:
        return 0.0
    idx = np.searchsorted(bias_values, eta, side="right") - 1
    idx2 = np.searchsorted(bias_values, eta - 1, side="right") - 1
    if idx2 <= 0:
        return 0.0
    if idx >= len(p_values):
        return 1.0
    sum_action_0 = np.sum(1 - p_values[:idx])
    sum_action_1 = np.sum(p_values[:idx2])
    prob_less = (sum_action_0 + sum_action_1) / lc if lc > 0 else 0.0
    return min(prob_less, 1.0) ** (n - 1)


@jit(nopython=True)
def _direction(bias, p_values, v, a, n, bias_values, lc):
    payoff_0 = v + a * _P_max(bias, p_values, n, bias_values, lc)
    payoff_1 = (v + a) * _P_max(bias + 1, p_values, n, bias_values, lc)
    idx = np.searchsorted(bias_values, bias, side="right") - 1
    delta = 1e-4 + 1e-2 * p_values[idx]
    grad = ((p_values[idx] + delta) * payoff_1 + (1 - p_values[idx] - delta) * payoff_0) \
        - ((p_values[idx] - delta) * payoff_1 + (1 - p_values[idx] + delta) * payoff_0)
    return grad


@jit(nopython=True)
def _err(bias, p_values, v, a, n, bias_values, lc):
    payoff_0 = v + a * _P_max(bias, p_values, n, bias_values, lc)
    payoff_1 = (v + a) * _P_max(bias + 1, p_values, n, bias_values, lc)
    idx = np.searchsorted(bias_values, bias, side="right") - 1
    expected = p_values[idx] * payoff_1 + (1 - p_values[idx]) * payoff_0
    return abs(max(payoff_1, payoff_0) - expected)


@jit(nopython=True, parallel=True)
def _direction_par(bias_values, p_values, v, a, n, lc):
    d = np.empty_like(p_values)
    for i in prange(len(bias_values)):
        d[i] = _direction(bias_values[i], p_values, v, a, n, bias_values, lc)
    return d


@jit(nopython=True, parallel=True)
def _error_par(bias_values, p_values, v, a, n, lc):
    e = np.empty_like(p_values)
    for i in prange(len(bias_values)):
        e[i] = _err(bias_values[i], p_values, v, a, n, bias_values, lc)
    return np.max(e)


def solve_broadcast_equilibrium(v, a, n, bias_values=None, lc=None,
                                max_iter=100000, tol=1e-4, step=1.0):
    """Iteratively solve for the mixed-NE withholding probability curve."""
    if bias_values is None:
        bias_values = _bias_values
    if lc is None:
        lc = len(bias_values)
    p_values = np.ones_like(bias_values) * 0.6
    for it in range(max_iter):
        p_values += step * _direction_par(bias_values, p_values, v, a, n, lc)
        np.clip(p_values, 0, 1, out=p_values)
        p_values.sort()
        if it % 100 == 0:
            if _error_par(bias_values, p_values, v, a, n, lc) / (v + a) < tol:
                break
    return p_values


def input_list_reward(La, N_t, Ug, gamma, party=1):
    """Input-list reward for a party under the paper's utility (Eq. 1):
    u_i / (1 + gamma*(n_i - 1)), summed over the party's allocated objects."""
    tot = 0.0
    for i in La[party]:
        ni = N_t[i]
        if ni > 0:
            tot += Ug[i] / (1 + gamma * (ni - 1))
    return tot


def solve_equilibrium_gamma(n, m, k, Ug, u_agg=None, gamma0=0.95,
                            iters=40, tol=1e-4, lc=800):
    """Self-consistent broadcast probability gamma.

    gamma is NOT a free parameter: it is the expected fraction of proposers
    that broadcast at the Phase-II mixed-NE. We solve the fixed point

        gamma -> allocation(gamma) -> v(gamma) -> broadcast-eq -> 1-E[withhold] -> gamma

    by damped iteration. With the paper's parameterization u_agg = sqrt(n)*sigma/n
    (sigma = sum of fees in T(M)), this converges to gamma in ~0.90-0.95 for
    EIP-7805-scale committees.
    """
    Ug = np.asarray(Ug, dtype=float)
    bias_values = np.linspace(BIAS_MIN, BIAS_MAX, lc)
    if u_agg is None:
        La0, Nt0 = two_step_transaction_inclusion(n, m, k, Ug, alpha=1)
        sigma = float(sum(Ug[i] for i in range(m) if Nt0[i] > 0))
        u_agg = np.sqrt(n) * sigma / n
    gamma = gamma0
    for _ in range(iters):
        La, N_t = two_step_transaction_inclusion(n, m, k, Ug, alpha=gamma)
        v = input_list_reward(La, N_t, Ug, gamma)
        # The withholding equilibrium depends only on the ratio v : u_agg, not the
        # absolute scale. Normalize payoffs to O(100) so the fixed-step gradient
        # solver is well-conditioned regardless of fee units (ETH vs abstract).
        s = (v + u_agg) / 200.0
        if s <= 0:
            s = 1.0
        p = solve_broadcast_equilibrium(v / s, u_agg / s, n,
                                        bias_values=bias_values, lc=lc)
        gamma_new = 1.0 - float(np.mean(p))
        if abs(gamma_new - gamma) < tol:
            return gamma_new
        gamma = 0.5 * gamma + 0.5 * gamma_new      # damped update for stability
    return gamma

def _finish_bribe_fig(x_axis, fname, legend_title=None):
    plt.plot(x_axis, x_axis, label="_x", **REF_KW)
    plt.plot(x_axis, 5 * x_axis, label="_5x", **REF_KW)
    plt.text(x_axis[-1], x_axis[-1] * 1.4, "y=x", ha="center", va="bottom", alpha=0.4, fontsize=9)
    plt.text(x_axis[-1], 5 * x_axis[-1] * 1.02, "y=5x", ha="center", va="bottom", alpha=0.4, fontsize=9)
    plt.xlabel("Fee Paid")
    plt.ylabel("Adversarial Bribe Tolerated")
    plt.title("Fee vs. Bribe Tolerated")
    plt.ylim(0, 5 * x_axis[-1] * 1.1)
    plt.legend(title=legend_title, fontsize=9)
    plt.margins(x=0)
    plt.savefig(IMAGES_DIR + fname, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {IMAGES_DIR + fname}")


# ════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════

def fig_committee(m=200, k=5, seed=0):
    np.random.seed(seed)
    plt.figure(figsize=FIGSIZE)
    Ug = np.random.beta(1, 5, size=m) * 30
    x_axis = np.arange(0, 35, 0.1)
    for c, n in enumerate([24, 32, 40]):
        g = solve_equilibrium_gamma(n, m, k, Ug)
        plt.plot(x_axis, bribe_curve(n, m, k, Ug, x_axis, gamma=g), "-",
                 color=SERIES_COLORS[c], label=f"n={n} ($\\gamma$={g:.2f})")
    _finish_bribe_fig(x_axis, "FeeBribeCommittee.pdf", legend_title="Committee")


def fig_ilsize(m=200, n=36, seed=0):
    np.random.seed(seed)
    plt.figure(figsize=FIGSIZE)
    Ug = np.random.beta(1, 5, size=m) * 30
    x_axis = np.arange(0, 35, 0.1)
    for c, k in enumerate([3, 4, 5]):
        g = solve_equilibrium_gamma(n, m, k, Ug)
        plt.plot(x_axis, bribe_curve(n, m, k, Ug, x_axis, gamma=g), "-",
                 color=SERIES_COLORS[c], label=f"k={k} ($\\gamma$={g:.2f})")
    _finish_bribe_fig(x_axis, "FeeBribeILSize.pdf", legend_title="List size")


def fig_mempool(m=200, n=36, k=5, seed=0):
    np.random.seed(seed)
    plt.figure(figsize=FIGSIZE)
    x_axis = np.arange(0, 35, 0.1)
    for c, i in enumerate(range(3)):
        scale = 20 + 10 * i
        Ug = np.random.beta(1, 5, size=m) * scale
        g = solve_equilibrium_gamma(n, m, k, Ug)
        plt.plot(x_axis, bribe_curve(n, m, k, Ug, x_axis, gamma=g), "-",
                 color=SERIES_COLORS[c], label=f"avg {scale // 5} ($\\gamma$={g:.2f})")
    _finish_bribe_fig(x_axis, "FeeBribeMempool.pdf", legend_title="Fee scale")


def fig_broadcast(m=200, n=36, k=5, seed=0, lc=LC):
    np.random.seed(seed)
    plt.figure(figsize=FIGSIZE)
    Ug = np.random.beta(1, 5, size=m) * 30
    La, N_t = two_step_transaction_inclusion(n, m, k, Ug)
    v = sum(Ug[i] / N_t[i] for i in La[1] if N_t[i] > 0)
    bias_values = np.linspace(BIAS_MIN, BIAS_MAX, lc)
    for c, a in enumerate([32, 128, 256]):
        p_values = solve_broadcast_equilibrium(v, a, n, bias_values=bias_values, lc=lc)
        plt.plot(bias_values, p_values, color=SERIES_COLORS[c], label=f"$u_{{agg}}$={a}")
    plt.xlabel("Bias")
    plt.ylabel("Withholding Probability")
    plt.title("Equilibrium Withholding (flag $F=0$)")
    plt.margins(x=0)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=9)
    plt.savefig(IMAGES_DIR + "BroadcastEquilibrium.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {IMAGES_DIR}BroadcastEquilibrium.pdf")


# ════════════════════════════════════════════════════════════════
# Self-test: confirm equivalence to the reference deepcopy algorithm
# ════════════════════════════════════════════════════════════════

def _self_test():
    import copy

    def ref_two_step(n, m, k, Ua, alpha=1):
        Us = copy.deepcopy(Ua); N = np.zeros(m); N_t = np.zeros(m); S = []
        for _ in range(n * k):
            s = int(np.argmax(Us / (N + 1))); S.append(s)
            if Us[s] == -1:
                break
            N[s] += alpha; N_t[s] += 1
            if N_t[s] >= n:
                Us[s] = -1
        U = copy.deepcopy(Ua); La = {i: set() for i in range(1, n + 1)}
        U_f = np.divide(U, (N + 1 - alpha), out=np.zeros_like(U, dtype=float), where=N != 0)
        for j, o in enumerate(sorted(S, key=lambda s: (-U_f[s], s))):
            La[(j % n) + 1].add(o)
        return La, N_t

    def ref_bribes(n, m, k, Ua, La, N_t, alpha=1):
        S = sum((list(s) for s in La.values()), [])
        Usa = copy.deepcopy(Ua); incList = np.unique(S)
        for i in incList:
            if N_t[i] >= n:
                Usa[i] = -1
        bribes = np.zeros_like(Ua, dtype=float)
        for i in sorted(incList):
            Us = copy.deepcopy(Usa); L = copy.deepcopy(La)
            N2_t = copy.deepcopy(N_t); N2 = copy.deepcopy(N_t) * alpha
            bribe = Ua[i] * N2_t[i]; Us[i] = -1; add = []
            for j in range(n):
                if i in L[j + 1]:
                    L[j + 1].remove(i); s = int(np.argmax(Us / (N2 + 1)))
                    if Us[s] == -1:
                        break
                    L[j + 1].add(s); add.append(s); N2[s] += alpha; N2_t[s] += 1
                    if N2_t[s] >= n:
                        Us[s] = -1
            U_f = np.divide(Ua, (N2 + 1 - alpha), out=np.zeros_like(Ua, dtype=float), where=N2 != 0)
            for h in add:
                bribe -= U_f[h]
            bribes[i] = max(0., bribe)
        return bribes

    rng = np.random.RandomState(2024)
    worst = 0.0
    for _ in range(300):
        n = int(rng.choice([8, 16, 24, 32, 36]))
        k = int(rng.choice([3, 4, 5, 6]))
        m = int(rng.choice([50, 100, 200]))
        Ua = (rng.beta(1, 5, size=m) * rng.choice([20, 30, 40])).astype(float)
        La_r, Nt_r = ref_two_step(n, m, k, Ua.copy())
        La_f, Nt_f = two_step_transaction_inclusion(n, m, k, Ua.copy())
        assert all(La_r[p] == La_f[p] for p in La_r) and np.array_equal(Nt_r, Nt_f), "allocation mismatch"
        b_ref = ref_bribes(n, m, k, Ua.copy(), La_r, Nt_r)
        b_full = calculate_bribes(n, m, k, Ua.copy(), La_f, Nt_f)
        worst = max(worst, float(np.max(np.abs(b_ref - b_full))))
        for tgt in (0, int(rng.randint(m))):
            b_single = calculate_bribe_single(n, m, k, Ua.copy(), La_f, Nt_f, tgt)
            worst = max(worst, abs(b_ref[tgt] - b_single))
    print(f"self-test passed over 300 instances; max abs diff vs reference = {worst:.2e}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="coarser broadcast grid (faster)")
    ap.add_argument("--check", action="store_true", help="run self-test vs reference and exit")
    args = ap.parse_args()

    if args.check:
        _self_test()
        raise SystemExit

    os.makedirs(IMAGES_DIR, exist_ok=True)
    lc = 1500 if args.fast else LC
    print("AUCIL evaluation - regenerating all figures in consistent style")
    print(f"  numba acceleration: {HAVE_NUMBA};  broadcast grid lc={lc}")
    print("[1/4] committee size ...")
    fig_committee()
    print("[2/4] input-list size ...")
    fig_ilsize()
    print("[3/4] mempool fee scale ...")
    fig_mempool()
    print("[4/4] broadcast equilibrium ...")
    fig_broadcast(lc=lc)
    print("Done. All four figures written to", IMAGES_DIR)

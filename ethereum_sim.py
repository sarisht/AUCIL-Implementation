#!/usr/bin/env python3
"""
AUCIL on a live Ethereum block
==============================
Fetches the latest Ethereum block, extracts each transaction's priority fee
(tip), and runs the AUCIL pipeline with the parameters we recommend for
Ethereum (EIP-7805 scale). It reports the censorship-resistance guarantees in
ETH and as multiples of the transaction fee.

Recommended Ethereum parameters (EIP-7805):
  n  = 16     committee size      (IL_COMMITTEE_SIZE = 2^4)
  k  = 5      input-list size     (~ a few average txns, <= 8 KiB)
  th = 4      crash tolerance     (25%)
  b_max = sqrt(n) = 4             VRF bias range
  u_agg = sqrt(n) * sigma / n     aggregation reward (sigma = tip mass in T(M))
  gamma = solve_equilibrium_gamma(...)   self-consistent broadcast prob (~0.91)

Usage:
  python3 ethereum_sim.py                  # live public RPC
  python3 ethereum_sim.py --rpc URL        # specific RPC endpoint
  python3 ethereum_sim.py --demo           # offline synthetic block
  python3 ethereum_sim.py --source dune --query-id 1234567   # Dune (needs DUNE_API_KEY)
  python3 ethereum_sim.py --dump-fees fees.txt               # dump fees for the explorer

Dune SQL to save as a query (latest-block per-tx effective tip, gwei):
  WITH latest AS (SELECT max(block_number) AS bn FROM ethereum.transactions)
  SELECT t.block_number,
         t.gas_used,
         (LEAST(t.max_priority_fee_per_gas,
                t.max_fee_per_gas - b.base_fee_per_gas)) / 1e9              AS tip_gwei,
         (LEAST(t.max_priority_fee_per_gas,
                t.max_fee_per_gas - b.base_fee_per_gas)) / 1e9 * t.gas_used AS tip_total_gwei
  FROM ethereum.transactions t
  JOIN ethereum.blocks b ON t.block_number = b.number
  JOIN latest            ON t.block_number = latest.bn
"""

import argparse
import json
import os
import ssl
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulations as ev   # verified algorithm + figure style

IMAGES_DIR = "Figures/"
SERIES_COLORS = ev.SERIES_COLORS
REF_KW = ev.REF_KW

PUBLIC_RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.drpc.org",
    "https://1rpc.io/eth",
    "https://eth.meowrpc.com",
]
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


# ════════════════════════════════════════════════════════════════
# Data acquisition
# ════════════════════════════════════════════════════════════════

def _rpc(url, method, params, timeout=30):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        out = json.loads(r.read())
    if "result" not in out or out["result"] is None:
        raise RuntimeError(out.get("error", "no result"))
    return out["result"]


def fetch_block_rpc(rpc_url=None):
    """Return (block_number, tips_eth ndarray, meta) from a public JSON-RPC node.

    Per-transaction tip = (effectiveGasPrice - baseFeePerGas) * gasUsed, in ETH.
    """
    urls = [rpc_url] if rpc_url else PUBLIC_RPCS
    last_err = None
    for url in urls:
        try:
            blk = _rpc(url, "eth_getBlockByNumber", ["latest", False])
            bn_hex = blk["number"]
            base = int(blk["baseFeePerGas"], 16)
            receipts = _rpc(url, "eth_getBlockReceipts", [bn_hex])
            tips = []
            for r in receipts:
                egp = int(r["effectiveGasPrice"], 16)
                gu = int(r["gasUsed"], 16)
                tip = max(egp - base, 0) * gu / 1e18      # ETH
                tips.append(tip)
            tips = np.array([t for t in tips if t > 0], dtype=float)
            meta = dict(source=f"RPC {url}", base_fee_gwei=base / 1e9,
                        n_tx=len(receipts), n_tip=len(tips))
            return int(bn_hex, 16), tips, meta
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all RPC endpoints failed (last: {last_err})")


def fetch_block_dune(query_id, api_key, execute=False):
    """Return (block_number, tips ndarray, meta) from a saved Dune query."""
    base = "https://api.dune.com/api/v1"
    hdr = {"X-Dune-API-Key": api_key, "User-Agent": _UA}

    def _get(path):
        req = urllib.request.Request(base + path, headers=hdr)
        with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
            return json.loads(r.read())

    def _post(path):
        req = urllib.request.Request(base + path, data=b"{}",
                                     headers={**hdr, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
            return json.loads(r.read())

    if execute:
        ex = _post(f"/query/{query_id}/execute")
        eid = ex["execution_id"]
        import time
        for _ in range(60):
            st = _get(f"/execution/{eid}/status")
            state = st.get("state")
            if state == "QUERY_STATE_COMPLETED":
                break
            if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                raise RuntimeError(f"Dune execution {state}")
            time.sleep(2)
        data = _get(f"/execution/{eid}/results")
    else:
        data = _get(f"/query/{query_id}/results")     # cached latest results (cheaper)

    rows = data["result"]["rows"]
    if not rows:
        raise RuntimeError("Dune query returned no rows")
    fee_col = _detect_fee_col(rows[0])
    tips = np.array([float(r[fee_col]) for r in rows if r.get(fee_col) not in (None, "")], dtype=float)
    tips = tips[tips > 0]
    bn = None
    for c in ("block_number", "number", "blockNumber"):
        if c in rows[0]:
            bn = int(rows[0][c]); break
    meta = dict(source=f"Dune query {query_id}", fee_col=fee_col, n_tx=len(rows), n_tip=len(tips))
    return bn, tips, meta


def _detect_fee_col(row):
    prefer = ["tip_total_gwei", "tip_gwei", "priority_fee_gwei", "priority_fee",
              "tip", "fee", "value"]
    for c in prefer:
        if c in row:
            return c
    for c, v in row.items():
        if c.lower() in ("block_number", "number", "gas_used", "gas", "nonce"):
            continue
        try:
            float(v); return c
        except (TypeError, ValueError):
            continue
    raise RuntimeError(f"could not detect a fee column in row keys {list(row)}")


def synthetic_block(seed=0):
    """Realistic offline block: ~180 txns, log-normal priority tips (ETH)."""
    rng = np.random.RandomState(seed)
    n_tx = rng.randint(150, 220)
    tip_gwei = rng.lognormal(mean=0.6, sigma=1.1, size=n_tx)
    gas = rng.lognormal(mean=11.3, sigma=0.7, size=n_tx)
    tips = tip_gwei * 1e-9 * gas      # ETH
    tips = tips[tips > 0]
    meta = dict(source="synthetic (offline demo)", base_fee_gwei=1.0, n_tx=n_tx, n_tip=len(tips))
    return None, np.array(tips, dtype=float), meta


# ════════════════════════════════════════════════════════════════
# AUCIL analysis with EIP-7805 parameters
# ════════════════════════════════════════════════════════════════

def run_aucil(tips, n=16, k=5, theta=4):
    """Run AUCIL on a vector of per-transaction tips (ETH). Returns a report dict."""
    Ug = np.asarray(tips, dtype=float)
    m = len(Ug)
    La1, Nt1 = ev.two_step_transaction_inclusion(n, m, k, Ug, alpha=1)
    sigma = float(sum(Ug[i] for i in range(m) if Nt1[i] > 0))      # tip mass in T(M)
    u_agg = np.sqrt(n) * sigma / n
    gamma = ev.solve_equilibrium_gamma(n, m, k, Ug, u_agg=u_agg)

    La, Nt = ev.two_step_transaction_inclusion(n, m, k, Ug, alpha=gamma)
    bribes = ev.calculate_bribes(n, m, k, Ug, La, Nt, alpha=gamma)
    selected = [i for i in range(m) if Nt[i] > 0]

    agg_cr = (n - theta) * u_agg / 2.0
    med_fee = float(np.median(Ug)) if m else 0.0
    med_sel_fee = float(np.median([Ug[i] for i in selected])) if selected else 0.0

    # Input CR is the binding layer (realized cost is min(beta1, beta2, beta3)),
    # so we summarize the per-transaction Input-CR multiple over the protected set.
    sel_ratios = [float(bribes[i] / Ug[i]) for i in selected if Ug[i] > 0 and bribes[i] > 0]
    input_cr_ratio_median = float(np.median(sel_ratios)) if sel_ratios else 0.0
    input_cr_ratio_max = float(np.max(sel_ratios)) if sel_ratios else 0.0
    # Among the protected transactions, those whose Input CR strictly exceeds their fee.
    n_above_fee = int(sum(1 for r in sel_ratios if r > 1.0))

    # Input-CR multiple for the highest-fee transactions: those at or above the
    # 95th fee percentile of the protected set. We report the range (min..max)
    # of the Input-CR/fee ratio over this top slice, since that is where the
    # protection is concentrated.
    p95_lo = p95_hi = 0.0
    p95_count = 0
    if selected:
        sel_fees = np.array([Ug[i] for i in selected])
        thresh95 = float(np.percentile(sel_fees, 95))
        top95 = [i for i in selected if Ug[i] >= thresh95 and Ug[i] > 0 and bribes[i] > 0]
        ratios95 = [float(bribes[i] / Ug[i]) for i in top95]
        if ratios95:
            p95_lo, p95_hi = float(min(ratios95)), float(max(ratios95))
            p95_count = len(ratios95)

    top = sorted(selected, key=lambda i: -Ug[i])[:8]
    top_rows = [dict(fee=float(Ug[i]), n_t=int(Nt[i]),
                     input_cr=float(bribes[i]),
                     ratio=(float(bribes[i] / Ug[i]) if Ug[i] > 0 else 0.0)) for i in top]

    return dict(n=n, k=k, theta=theta, m=m, gamma=gamma, sigma=sigma, u_agg=u_agg,
                n_selected=len(selected), agg_cr=agg_cr,
                med_fee=med_fee, med_sel_fee=med_sel_fee,
                agg_cr_x_med_sel=(agg_cr / med_sel_fee if med_sel_fee else 0.0),
                agg_cr_x_med_all=(agg_cr / med_fee if med_fee else 0.0),
                input_cr_ratio_median=input_cr_ratio_median,
                input_cr_ratio_max=input_cr_ratio_max,
                n_above_fee=n_above_fee,
                p95_lo=p95_lo, p95_hi=p95_hi, p95_count=p95_count,
                top=top_rows, Ug=Ug, La=La, Nt=Nt, bribes=bribes, selected=selected)


# ════════════════════════════════════════════════════════════════
# Reporting + figure
# ════════════════════════════════════════════════════════════════

def print_report(block_number, meta, rep):
    line = "=" * 66
    print(line)
    print("  AUCIL on an Ethereum block")
    print(line)
    print(f"  data source       : {meta['source']}")
    if block_number is not None:
        print(f"  block number      : {block_number:,}")
    print(f"  transactions      : {meta.get('n_tip', rep['m'])} with positive tip")
    print(f"  parameters        : n={rep['n']}, k={rep['k']}, theta={rep['theta']}, "
          f"b_max=sqrt(n)={int(np.sqrt(rep['n']))}")
    print(f"  equilibrium gamma : {rep['gamma']:.3f}   (self-consistent broadcast prob)")
    print(line)
    print(f"  T(M) size         : {rep['n_selected']} transactions selected into the IL")
    print(f"  tip mass sigma    : {rep['sigma']:.6f} ETH")
    print(f"  aggregation reward: u_agg = {rep['u_agg']:.6f} ETH")
    print()
    print(f"  Input CR for the highest-fee transactions (the binding layer):")
    print(f"      {'fee (ETH)':>13} {'n_t':>4} {'input CR (ETH)':>16} {'xfee':>7}")
    for r in rep["top"]:
        print(f"      {r['fee']:>13.6f} {r['n_t']:>4d} {r['input_cr']:>16.6f} "
              f"{(str(round(r['ratio'],2))+'x') if r['ratio']>0 else '-':>7}")
    print()
    print(f"  Protected transactions with Input CR > fee: {rep['n_above_fee']} of {rep['n_selected']}")
    print(f"  Input-CR multiple over protected set: median {rep['input_cr_ratio_median']:.1f}x, "
          f"max {rep['input_cr_ratio_max']:.1f}x")
    print(f"  Input-CR multiple for top fees (>=95th pct, {rep['p95_count']} txns): "
          f"{rep['p95_lo']:.1f}x to {rep['p95_hi']:.1f}x")
    print(line)
    print(f"  Headline: under the current system, censoring a transaction costs")
    print(f"  ~1x its fee (bribe the sole builder). Under AUCIL the realized cost is")
    print(f"  min(Input, Aggregation, Blockchain) CR; the binding layer is Input CR.")
    print(f"  For transactions paying above the 95th fee percentile, Input CR is")
    print(f"  {rep['p95_lo']:.0f}x-{rep['p95_hi']:.0f}x the transaction's own fee. Aggregation CR "
          f"({rep['agg_cr']:.4f} ETH)")
    print(f"  is strictly higher, so it is never the bottleneck.")
    print(line)


def _label_guides(xhi, yhi):
    """Annotate the y=x and y=5x reference lines just inside the right edge.

    Right-aligned at the x upper limit so each label sits on its guide line and
    stays within the axes -- previously the labels were centered on the boundary
    and spilled outside the plot. Shared by both Ethereum figures so they look
    identical.
    """
    xlab = 0.98 * xhi
    plt.text(xlab, xlab, "y=x", ha="right", va="bottom", alpha=0.4, fontsize=9)
    plt.text(xlab, min(5 * xlab, 0.97 * yhi), "y=5x",
             ha="right", va="bottom", alpha=0.4, fontsize=9)


def save_figure(blocks):
    """Scatter Input CR vs. fee for every protected tx, overlaying each block.

    Real-block fees are heavy-tailed: most protected txs pay a small fee and get
    little protection, while a few high-fee txs carry the large (10-15x) Input CR
    that is the whole point. On linear axes that tail is either clipped or
    squashed into a corner, wasting most of the plot -- so we use log-log axes:
    every point stays visible and the y=x / y=5x guides remain straight, parallel
    reference lines. ``blocks`` is a list of (block_number, report) pairs.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    scale = 1e9                       # ETH -> Gwei-ETH for readable tick labels
    plt.figure(figsize=ev.FIGSIZE)
    fxs, bys, dropped = [], [], 0
    for j, (block_number, rep) in enumerate(blocks):
        Ug, bribes, selected = rep["Ug"], rep["bribes"], rep["selected"]
        fx = np.array([Ug[i] for i in selected]) * scale
        by = np.array([bribes[i] for i in selected]) * scale
        pos = by > 0                  # log axes can't show zero-CR marginal txs
        dropped += int((~pos).sum())
        fx, by = fx[pos], by[pos]
        if len(fx) == 0:
            continue
        plt.scatter(fx, by, s=18, alpha=0.75,
                    color=ev.SERIES_COLORS[j % len(ev.SERIES_COLORS)],
                    label=f"block {block_number}" if block_number else f"block {j + 1}")
        fxs.append(fx)
        bys.append(by)
    if not fxs:
        print("  (no protected transactions; skipping figure)")
        plt.close()
        return
    fx_all, by_all = np.concatenate(fxs), np.concatenate(bys)
    xlo, xhi = fx_all.min() / 1.5, fx_all.max() * 1.5
    # The y range must hold both the data and the guide endpoints over [xlo, xhi].
    ylo = min(by_all.min(), xlo) / 1.5
    yhi = max(by_all.max(), 5 * xhi) * 1.5
    xr = np.array([xlo, xhi])
    plt.plot(xr, xr, label="_x", **ev.REF_KW)
    plt.plot(xr, 5 * xr, label="_5x", **ev.REF_KW)
    xg = xhi / 1.3                     # label the guides just inside the right edge
    plt.text(xg, xg, "y=x", ha="right", va="bottom", alpha=0.4, fontsize=9)
    plt.text(xg, 5 * xg, "y=5x", ha="right", va="bottom", alpha=0.4, fontsize=9)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel("Fee Paid")
    plt.ylabel("Adversarial Bribe Tolerated")
    plt.title("Fee vs. Bribe Tolerated")
    plt.legend(fontsize=8)
    plt.savefig(IMAGES_DIR + "EthereumBlockCR.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    if dropped:
        print(f"  (note: {dropped} marginal txs with zero Input CR omitted from log plot)")
    print(f"  figure written    : {IMAGES_DIR}EthereumBlockCR.pdf")


def fee_sweep(blocks, n=16, k=5, theta=4, npts=40):
    """Insert one of OUR transactions into each block and sweep its fee.

    For every block we append a target transaction to the real per-tx tip
    vector, vary its fee from 0 to ~1.5x that block's current max tip, and
    compute the target's Input CR (the binding layer) at each fee. Overlaying the
    blocks shows the protection-vs-fee curve is consistent across blocks.
    ``blocks`` is a list of (block_number, tips) pairs.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    scale = 1e9
    plt.figure(figsize=ev.FIGSIZE)
    xhi = yhi = 0.0
    for j, (block_number, tips) in enumerate(blocks):
        base = np.asarray(tips, dtype=float)
        target = len(base)                 # our tx is the last index
        Ug = np.append(base, 0.0)
        m = len(Ug)
        gamma = ev.solve_equilibrium_gamma(n, m, k, Ug)

        xmax = float(base.max()) * 1.5
        xs = np.linspace(0.0, xmax, npts)
        ys = ev.bribe_curve(n, m, k, Ug, xs, target=target, gamma=gamma)
        fx, by = xs * scale, np.asarray(ys) * scale
        plt.plot(fx, by, linewidth=2,
                 color=ev.SERIES_COLORS[j % len(ev.SERIES_COLORS)],
                 label=f"block {block_number}" if block_number else f"block {j + 1}")
        xhi = max(xhi, float(fx.max()))
        yhi = max(yhi, float(by.max()))

        # console reference points for the first (representative) block
        if j == 0:
            print(f"  fee sweep (our tx) : block {block_number}, fee 0 -> {xmax*1e9:.2f} Gwei-gas")
            for frac in (0.25, 0.5, 1.0):
                gfee = xmax * frac
                b = float(ev.bribe_curve(n, m, k, Ug, np.array([gfee]), target=target, gamma=gamma)[0])
                mult = (b / gfee) if gfee > 0 else 0.0
                print(f"      fee={gfee*1e9:8.2f} Gwei-gas  Input CR={b*1e9:8.2f}  ({mult:.1f}x)")

    yhi = max(yhi, 5 * xhi) * 1.1
    xr = np.linspace(0, xhi, 50)
    plt.plot(xr, xr, label="_x", **ev.REF_KW)
    plt.plot(xr, 5 * xr, label="_5x", **ev.REF_KW)
    _label_guides(xhi, yhi)
    plt.xlabel("Fee Paid")
    plt.ylabel("Adversarial Bribe Tolerated")
    plt.title("Fee vs. Bribe Tolerated")
    plt.legend(fontsize=8)
    plt.xlim(0, xhi)
    plt.ylim(0, yhi)
    plt.margins(x=0)
    plt.savefig(IMAGES_DIR + "EthereumFeeSweep.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  figure written    : {IMAGES_DIR}EthereumFeeSweep.pdf")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def load_pinned_block(path="pinned_block.txt"):
    """Load the committed representative block (per-tx tips in ETH).

    This makes the figures reproducible: by default we use this pinned block
    rather than whatever block is live at run time. The block was captured from
    a live public RPC; its number is recorded in the file header.
    """
    bn = None
    tips = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if "block_number=" in line:
                    bn = int(line.split("block_number=")[1].strip())
                continue
            tips = [float(x) for x in line.strip().split(",") if x]
    tips = np.array([t for t in tips if t > 0], dtype=float)
    meta = dict(source=f"pinned block {bn}", n_tx=len(tips), n_tip=len(tips))
    return bn, tips, meta


def load_pinned_blocks(path="pinned_blocks.txt"):
    """Load the committed representative blocks (per-tx tips in ETH).

    Multi-block companion to load_pinned_block: the file holds several stanzas,
    each a ``# block_number=N`` header followed by one CSV line of positive
    per-tx tips. Returns a list of (block_number, tips, meta).
    """
    blocks, bn, tips = [], None, None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if "block_number=" in line:
                    if bn is not None and tips is not None:
                        blocks.append((bn, tips))
                    bn = int(line.split("block_number=")[1].strip())
                    tips = None
                continue
            vals = [float(x) for x in line.strip().split(",") if x]
            tips = np.array([t for t in vals if t > 0], dtype=float)
    if bn is not None and tips is not None:
        blocks.append((bn, tips))
    return [(b, t, dict(source=f"pinned block {b}", n_tx=len(t), n_tip=len(t)))
            for b, t in blocks]


def average_blocks(n_blocks=10, n=16, k=5, theta=4, rpc=None):
    """Fetch n_blocks live blocks and average the headline Input-CR metrics.

    Intended to be run locally (needs ~n_blocks live RPC round-trips). Prints a
    table and the mean 95th-percentile Input-CR range across blocks.
    """
    rows, p95los, p95his, maxes = [], [], [], []
    bn0 = None
    for _ in range(n_blocks):
        try:
            bn, tips, meta = fetch_block_rpc(rpc)
        except Exception as e:
            print(f"  [warn] fetch failed ({e}); skipping")
            continue
        rep = run_aucil(tips, n=n, k=k, theta=theta)
        rows.append((bn, meta.get("n_tip"), rep["n_selected"],
                     rep["p95_lo"], rep["p95_hi"], rep["input_cr_ratio_max"]))
        if rep["p95_lo"] > 0:
            p95los.append(rep["p95_lo"]); p95his.append(rep["p95_hi"])
        maxes.append(rep["input_cr_ratio_max"])
    print(f"{'block':>10} {'txns':>5} {'T(M)':>5} {'p95_lo':>7} {'p95_hi':>7} {'max':>6}")
    for r in rows:
        print(f"{r[0]:>10} {r[1]:>5} {r[2]:>5} {r[3]:>6.1f}x {r[4]:>6.1f}x {r[5]:>5.1f}x")
    if p95los:
        print(f"  mean over {len(rows)} blocks: 95th-pct Input CR "
              f"{np.mean(p95los):.1f}x to {np.mean(p95his):.1f}x; "
              f"mean max {np.mean(maxes):.1f}x")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Run AUCIL on a representative (pinned) or live Ethereum block.")
    ap.add_argument("--source", choices=["pinned", "rpc", "dune"], default="pinned",
                    help="pinned = committed reproducible block (default); rpc = live; dune = Dune query")
    ap.add_argument("--rpc", default=None, help="specific JSON-RPC endpoint")
    ap.add_argument("--query-id", default=None, help="Dune saved-query id (with --source dune)")
    ap.add_argument("--api-key", default=None, help="Dune API key (else env DUNE_API_KEY)")
    ap.add_argument("--execute", action="store_true", help="re-execute the Dune query (costs credits)")
    ap.add_argument("--demo", action="store_true", help="use an offline synthetic block")
    ap.add_argument("--blocks", type=int, default=0,
                    help="average headline metrics over this many live blocks (run locally), then exit")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--theta", type=int, default=4)
    ap.add_argument("--dump-fees", default=None, help="write fetched tips (comma list) to a file")
    args = ap.parse_args()

    if args.blocks > 0:
        average_blocks(args.blocks, n=args.n, k=args.k, theta=args.theta, rpc=args.rpc)
        return

    if args.demo:
        acquired = [synthetic_block()]
    elif args.source == "dune":
        key = args.api_key or os.environ.get("DUNE_API_KEY")
        if not key:
            raise SystemExit("Dune source needs --api-key or env DUNE_API_KEY")
        if not args.query_id:
            raise SystemExit("Dune source needs --query-id (see SQL in this file's docstring)")
        acquired = [fetch_block_dune(args.query_id, key, execute=args.execute)]
    elif args.source == "rpc":
        try:
            acquired = [fetch_block_rpc(args.rpc)]
        except Exception as e:
            print(f"  [warn] live fetch failed ({e}); falling back to pinned blocks.")
            acquired = load_pinned_blocks()
    else:
        acquired = load_pinned_blocks()

    # Run AUCIL on each acquired block (the pinned default has three; live/demo
    # sources have one, so the figures simply show a single series).
    runs = []
    for block_number, tips, meta in acquired:
        if tips is None or len(tips) == 0:
            continue
        rep = run_aucil(tips, n=args.n, k=args.k, theta=args.theta)
        runs.append((block_number, tips, meta, rep))
    if not runs:
        raise SystemExit("no transactions with positive tip found")

    if args.dump_fees:
        tips0 = runs[0][1]
        vals = ", ".join(f"{t*1e9:.4f}" for t in tips0)     # Gwei-ETH units
        with open(args.dump_fees, "w") as f:
            f.write(vals)
        print(f"  fees dumped to    : {args.dump_fees}  ({len(tips0)} values)")

    # Report the first (representative) block in detail; figures overlay all.
    bn0, tips0, meta0, rep0 = runs[0]
    print_report(bn0, meta0, rep0)
    save_figure([(bn, rep) for bn, _, _, rep in runs])
    fee_sweep([(bn, tips) for bn, tips, _, _ in runs], n=args.n, k=args.k, theta=args.theta)


if __name__ == "__main__":
    main()

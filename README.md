# AUCIL-Implementation

Artifact for **AUCIL: An Inclusion List Design for Rational Parties**.

## Contents

- `simulations.py` — core AUCIL algorithm (input selection, VRF-biased bidding,
  aggregation, bribe/Input-CR computation)
  (`FeeBribeCommittee`, `FeeBribeILSize`, `FeeBribeMempool`, `BroadcastEquilibrium`).
- `ethereum_sim.py` — runs the full AUCIL pipeline on real Ethereum blocks and
  produces the censorship-resistance figures (`EthereumBlockCR`, `EthereumFeeSweep`).
- `pinned_blocks.txt` — three committed Ethereum blocks (per-tx tips), so the
  Ethereum figures are reproducible offline.
- `algorithm_explorer.html` — interactive, self-contained explorer for Algorithm 1.
  Open it in any browser.
- `Artifact for proofs in AUCIL.pdf` — appendices and proofs.

## Setup

```sh
sudo apt-get update
sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended
pip install numpy matplotlib numba   # numba is optional (speeds up the solver)
```

## Ethereum data (optional, for live runs)

By default `ethereum_sim.py` uses the committed `pinned_blocks.txt`. To run on a
live block instead, use a public RPC (`--source rpc`) or a Dune query
(`--source dune --query-id <ID>`, with `DUNE_API_KEY` set). The Dune query below
returns one row per transaction of the latest block (effective priority tip);
save it and pass its id:

```sql
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
```

## Reproducing the figures

```sh
python3 simulations.py                # parameter-sweep figures -> Figures/
python3 ethereum_sim.py               # Ethereum figures from the 3 pinned blocks -> Figures/
python3 ethereum_sim.py --source rpc  # optional: use a live block instead of the pinned set
```

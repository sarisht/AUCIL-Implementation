# AUCIL-Implementation

Artifact for **AUCIL: An Inclusion List Design for Rational Parties**.

## Contents

- `simulations.py` — core AUCIL algorithm (input selection, VRF-biased bidding,
  aggregation, bribe/Input-CR computation) and the parameter-sweep figures
  (`FeeBribeCommittee`, `FeeBribeILSize`, `FeeBribeMempool`, `BroadcastEquilibrium`).
- `ethereum_sim.py` — runs the full AUCIL pipeline on real Ethereum blocks and
  produces the censorship-resistance figures (`EthereumBlockCR`, `EthereumFeeSweep`).
  Imports `simulations.py` for the verified algorithm.
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

## Reproducing the figures

```sh
python3 simulations.py                # parameter-sweep figures -> Figures/
python3 ethereum_sim.py               # Ethereum figures from the 3 pinned blocks -> Figures/
python3 ethereum_sim.py --source rpc  # optional: use a live block instead of the pinned set
```

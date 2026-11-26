<div align="center">

<h2>Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization</h2>

<a href='https://icml.cc/virtual/2026/oral/71051'><img src='https://img.shields.io/badge/ICML-2026%20Oral-blue'></a>
<a href='https://arxiv.org/abs/2506.05747'><img src='https://img.shields.io/badge/arXiv-2506.05747-b31b1b'></a>


![Strategy trajectories on biased RPS](assets/brps_x_ternary.gif)



</div>

## Overview

In saddle-point (minimax) optimization, gradient descent ascent (**GDA**) often **cycles instead of converging**.
A widely used **symmetric payoff perturbation** penalizes both players' payoffs to stabilize the learning dynamics, yielding **SymP-GDA**.
However, it shifts the equilibrium away from the original game, so its strength must be annealed on a schedule, slowing convergence.

We propose **asymmetric perturbation**, which penalizes only *one* player's payoff.
For a small enough strength, the perturbed equilibrium strategy **exactly coincides** with an equilibrium strategy of the original game.
This yields **AsymP-GDA** with a **linear last-iterate convergence rate**, without any annealing schedule.
On normal-form and extensive-form benchmark games, AsymP-GDA converges fast to an equilibrium strategy in the original game.

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/). After [installing uv](https://docs.astral.sh/uv/getting-started/installation/), set up the environment with:

```bash
uv sync --locked
```

All commands below are run through `uv run`, which automatically uses the synced environment.

## Repository Structure

| Path | Description |
| --- | --- |
| `nfg/` | Normal-form game experiments (mirror-descent–based learners) |
| `efg/` | Extensive-form game experiments (dilated mirror descent and CFR baselines) |
| `notebooks/` | Notebooks for visualizing strategy trajectories |

---

## Normal-Form Games (NFG)

Run from the `nfg/` directory:

```bash
cd nfg
uv run main.py game=<GAME> \
  algorithm@player0=<ALG> player0.learning_rate=<LR> [player0.perturbation_strength=<MU>] \
  algorithm@player1=<ALG> player1.learning_rate=<LR> [player1.perturbation_strength=<MU>]
```

Choose the options below:

- `<GAME>`: `biased_rps` (biased Rock-Paper-Scissors, BRPS), `biased_matching_pennies` (biased matching pennies, BMP), or `m_ne` (M-Ne, a game with multiple Nash equilibria)
- `<ALG>` (per player):
  - `PGD` — perturbed gradient descent (the building block of **SymP-GDA** / **AsymP-GDA**; perturbs the player's payoff with the squared Euclidean regularizer)
  - `GD` / `MWU` — gradient descent / multiplicative weights update
  - `OGD` / `OMWU` — optimistic gradient descent / optimistic MWU
- `<MU>`: perturbation strength μ (only for `PGD`)
- `asymmetric_players=true` (default) lets the two players use different algorithms; set `asymmetric_players=false` to apply `player0`'s configuration symmetrically.

In the paper's terms, **AsymP-GDA** perturbs only one player: assign `PGD` to one player and `GD` to the other. **SymP-GDA** perturbs both players: use `PGD` with `asymmetric_players=false`. For example, AsymP-GDA:

```bash
uv run main.py game=biased_rps \
  algorithm@player0=PGD player0.perturbation_strength=1.0 player0.learning_rate=0.01 \
  algorithm@player1=GD   player1.learning_rate=0.01
```

### Reproducing the paper

Convenience scripts live in `nfg/scripts/`:

```bash
cd nfg
bash scripts/run_brps_comparison.sh          # NashConv vs iterations: AsymP-GDA, SymP-GDA, GDA, OGDA (biased RPS)
bash scripts/run_m_ne_comparison.sh          # NashConv vs iterations: AsymP-GDA, SymP-GDA, GDA, OGDA (M-Ne)
bash scripts/run_mu_sensitivity_asym_pgda.sh # NashConv vs iterations across μ: AsymP-GDA (biased RPS & M-Ne)
bash scripts/run_brps_vs_mu_asym_pgda.sh     # Proximity of x^μ to x* vs μ: AsymP-GDA (biased RPS)
bash scripts/run_brps_vs_mu_sym_pgda.sh      # Proximity of x^μ to x* vs μ: SymP-GDA (biased RPS)
bash scripts/run_bmp_vs_mu_asym_pgda.sh      # Proximity of x^μ to x* vs μ: AsymP-GDA (biased matching pennies)
bash scripts/run_bmp_vs_mu_sym_pgda.sh       # Proximity of x^μ to x* vs μ: SymP-GDA (biased matching pennies)
```

Results are written under `nfg/logs/`.

### Strategy trajectories

To dump strategy trajectories on biased RPS (used for the simplex/ternary plots):

```bash
cd nfg
bash scripts/run_brps_trajectory.sh
# or directly:
uv run scripts/dump_brps_trajectory.py --num-steps 5000 --learning-rate 0.002 \
  --mus 1.5 --algorithms gda symp_gda asymp_gda
```

The trajectories can be visualized with `notebooks/brps_ternary.ipynb`.
The notebooks need extra dependencies that a plain `uv sync` does not install,
so sync the `notebook` group and launch JupyterLab through it:

```bash
uv sync --group notebook
uv run --group notebook jupyter lab
```

---

## Extensive-Form Games (EFG)

Built on [OpenSpiel](https://github.com/google-deepmind/open_spiel). Run from the `efg/` directory:

```bash
cd efg
uv run main.py game=<GAME> algorithm=<ALG> iter=<ITER> [algorithm.eta=<ETA>] [algorithm.mu=<MU>]
```

Choose the options below:

- `<GAME>`: any OpenSpiel game string, e.g. `kuhn_poker`, `leduc_poker`, `"liars_dice(dice_sides=4)"`, or
  `"turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=5,points_order=descending))"`
- `<ALG>`:
  - `AsymPDGDA` — **AsymP-DGDA**: asymmetric perturbed dilated gradient descent ascent (ours)
  - `SymPDGDA` — **SymP-DGDA**: symmetric perturbed dilated gradient descent ascent
  - `DMWU` / `DGDA` — dilated mirror descent with the entropy / Euclidean regularizer
  - `DOMWU` / `DOGDA` — dilated optimistic mirror descent with the entropy / Euclidean regularizer
  - `CFR`, `CFRPlus`, `DCFR`, `LCFR` — CFR-family baselines
- `algorithm.eta`: learning rate η (mirror-descent methods)
- `algorithm.mu`: perturbation strength μ (`SymPDGDA`, `AsymPDGDA`)

Example (asymmetric perturbation on Kuhn poker):

```bash
uv run main.py game=kuhn_poker algorithm=AsymPDGDA algorithm.eta=0.1 algorithm.mu=0.01
```

### Reproducing the paper

One script per game runs every algorithm with the best hyperparameters reported in the paper (Table 4):

```bash
cd efg
bash scripts/run_kuhn_poker.sh        # Kuhn poker
bash scripts/run_leduc_poker.sh       # Leduc poker
bash scripts/run_liars_dice_4sides.sh # Liar's dice (4 sides)
bash scripts/run_goofspiel_4cards.sh  # Goofspiel (4 cards)
bash scripts/run_goofspiel_5cards.sh  # Goofspiel (5 cards)
```

Each script runs the six main-comparison methods (`DGDA`, `DMWU`, `DOGDA`, `DOMWU`, `SymPDGDA`, `AsymPDGDA`) together with the CFR-family baselines (`CFR`, `CFRPlus`, `DCFR`, `LCFR`). The config names follow the paper (`SymPDGDA` = **SymP-DGDA**, `AsymPDGDA` = **AsymP-DGDA**; also noted at the top of each script). Adjust `ITER` / `PRINT_FREQ` there if needed. Results are written under `efg/logs/`.

> **Note:** every reproduction script launches its algorithms as parallel background jobs, so they can consume a large amount of memory (the larger games use the most). To reduce peak memory, run fewer algorithms at a time by editing the script.

---

## Citation

Kenshi Abe, Mitsuki Sakamoto, Kaito Ariu, and Atsushi Iwasaki. Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization. In ICML, 2026.

Bibtex:
```
@inproceedings{abe2026asymmetric,
  title={Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization},
  author={Kenshi Abe and Mitsuki Sakamoto and Kaito Ariu and Atsushi Iwasaki},
  booktitle={ICML},
  year={2026},
}
```

## License

This project is released under the [MIT License](LICENSE).

"""Frozen synthetic DGP entry point; generation is offline only.
The byte-exact authors' functions live in sibling generator.R. Invoke Rscript
with source(generator.R) and simulate_dataset_design_based(N, K, .71, .85, seed).
N is an observed season match-count matrix. Inputs retain only W; partitions,
strengths and generation metadata are stored separately as labels/audit.
"""
from pathlib import Path
GENERATOR_R = Path(__file__).with_suffix('.R')

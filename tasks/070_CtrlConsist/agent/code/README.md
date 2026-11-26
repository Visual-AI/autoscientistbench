# Control Consistency Losses for Diffusion Bridges
### Samuel Howard, Nikolas Nüsken, Jakiw Pidstrigach
### ICML 2026 (Spotlight)

This respository contains code for [Control Consistency Losses for Diffusion Bridges](https://arxiv.org/abs/2512.05070), by Samuel Howard, Nikolas Nüsken and Jakiw Pidstrigach, presented at ICML 2026.
A earlier version of this work was presented at the Frontiers in Probabilistic Inference: Sampling Meets Learning Workshop at NeurIPS 2025 (Oral).

### Usage
Single training run:
`python train_bridge.py --config configs/ou.yaml --gpu 0 --train.lr 3e-4 --train.num_outer_iterations 5000`

Repeated runs:
`python run_repeats.py --config configs/ou.yaml --gpu 0 --num_repeats 5 --train.lr 3e-4 --train.num_outer_iterations 5000`

Benchmark entry points and the frozen protocol are documented in the parent README.

### Citation
If you find our paper or code useful, please consider citing as
<pre><code> @inproceedings{
howard2026control,
title={Control Consistency Losses for Diffusion Bridges},
author={Samuel Howard and Nikolas N{\"u}sken and Jakiw Pidstrigach},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}  </code></pre>
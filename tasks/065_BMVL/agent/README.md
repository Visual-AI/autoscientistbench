# Bootstrapping Multi-view Learning for Test-time Noisy Correspondence

## 1. Goal

Improve classification accuracy when some views of a multi-view example have incorrect correspondence. The fixed experiment uses five scored datasets held by the trusted evaluator, the original ten partitions and original 50% test corruption. The score is equal-dataset mean accuracy in percent, subject to the clean-condition mean accuracy being at least 91.80.

## 2. Data

The five scored datasets are held by the trusted evaluator. Because the original training partitions overlap other partitions' test rows, real training happens inside a separate private worker for each fold. Each worker receives only that fold's permitted data. Original per-view global min-max normalization, float32 conversion, stratified 80:20 partitions, row order and corruption are preserved.

`data/train/README.txt` explains the private fit interface. `data/val/input.json` and `data/val/private/` provide the **development set: the real CCV dataset** (6773 rows, 3 views, 20 classes), with seeds 0 to 4 of the same stratified 80:20 partition, the same global min-max normalization and the same 50% corruption. CCV shares no rows with the five scored datasets. `bench eval-dev` fits and scores it through the same private interfaces as the trusted evaluator, in about 4 minutes at 32 CPUs. The recorded CCV reference achieved `acc_eta50=52.396` and clean mean 60.282. Its cumulative private worker time was 1145.93 seconds, summed across parallel fit and inference calls; this is not wall time. The complete five-dataset population is always used for formal scoring.

## 3. Metric and scoring

For each dataset and original seed, compute accuracy ×100 and round to two decimals using NumPy. Average the ten values within each dataset, then average the five dataset means equally. `acc_eta50` is this noisy mean if the analogous clean mean is at least 91.80; otherwise it is exactly 0. On CCV the same rule uses five seeds and a clean floor of 58.00.

The frozen scorer verifies the complete population and class predictions. Private runtime isolation enforces fold-local access and separate clean/noisy inference. Malformed artifacts, execution failures and invalid predictions fail evaluation; their private logs are not returned. The clean accuracy guard is an ordinary score guard, not a format error.

Higher `acc_eta50` is better. For candidate mean C, archived reference mean R and reference standard deviation S, the relative gain `(C - R) / abs(R)` must be at least `max(0.005, 2 * S / abs(R))`. CCV is a harder dataset than the five scored ones, so compare a change on CCV with the CCV baseline from `bench status`, not with accuracy on other datasets.

## 4. Baseline

The baseline is the authors' BML implementation at upstream commit `49f8652da2a1dd5b97aabca4f745deda62e4ccab`. `code/multi_view.py` retains the original model and augmentation. `code/bridge.py` supplies private data entry, fitting and inference.

Default configuration remains 200 epochs, Adam with learning rate 0.002, batch size 2048, cosine scheduling to 1e-8, training correspondence augmentation 0.5, alignment-loss weight 1 and feature dimension 128. Each view uses the original three 1024-unit hidden layers. Training reports the final epoch without test-based selection. All real training is timed as private scoring work; outer `train.sh` prepares configuration only.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes candidate configuration and any public preparation artifacts.
- `bash code/predict.sh --model DIR --input PATH --output FILE` exports an inert `private-policy-v2` ZIP containing candidate code, prepared artifacts and any changed pinned Python dependencies.
- `code/policy.sh --model DIR --input DIR --output FILE` executes one private request. Fit requests contain only one training partition; inference requests contain that fit's opaque artifact and only one observed test condition. See `bench/io_spec.md`.
- `BENCH_CPUS` is the worker CPU allocation. `BENCH_SEED` is the outer invocation seed; the baseline uses the request's original `fit_seed` for the fixed ten initialization/split repetitions.
- `bench eval-dev` fits and scores the CCV development set. Submit code changes using `bench submit` before the overall deadline.

Dependencies are pinned in `code/env.lock` for Python 3.11, including the original PyTorch 2.1.2 CPU build, NumPy 1.26.4, SciPy 1.15.3 and scikit-learn 1.6.1. When `code/env.lock` exactly matches the frozen reference lock, private evaluation reuses those installed reference packages. Changed locks export their pinned packages in the candidate artifact.

## Research contract

**Objective.** Learn classification from the fixed multi-view datasets when view correspondences are noisy.

**Fixed research objects.** The five datasets, original ten partitions, normalization, test corruption, clean guard, accuracy aggregation and permitted information remain fixed. Supplied features were min-max normalized once before partitioning; this is part of the fixed observations, not a fitted step available to candidates. These define the shared classification comparison. Each noisy inference process receives only its noisy condition; clean inference is independently isolated.

**Editable methods.** You may replace the multi-view model, representation, training surrogate losses, optimizer and training schedule within this classification problem. Hyperparameters may be retuned as part of such a method change. Training augmentation derived from the authorized training fold is allowed.

**Forbidden substitutions.** Do not replace the datasets or noisy-correspondence problem, change accuracy aggregation or its guard, reconstruct hidden targets, use other folds' state, provide clean correspondence at noisy inference, acquire external data, embed released results or train on hidden labels. Model fitting and learned preprocessing must follow the Validation use section below. When retaining the supplied BML pipeline, changing only epochs=200, batch_size=2048, learning_rate=0.002, cosine eta_min=1e-8, augment_ratio=0.5, lambda_w=1, feat_dim=128, eps=1e-8 or dropout=0.5 does not count as a method improvement.

`bench/`, `data/`, this README and the external script signatures are frozen. Resources are 32 CPUs and 32 GB memory, bounded by the overall agent deadline and separate overall formal replay budget. A valid explicit submission is required.

## 7. Known pitfalls

CCV is not one of the five scored datasets, so an improvement on it is evidence of progress on the benchmark, not proof. Normal `predict.sh` exports an artifact rather than precomputing answers; the trusted scorer performs the real fits. Model state remains private and is never returned after a real fold. Each row has one class target shared by its views. For each selected row, the frozen corruption retains half the views, rounded up. Within each remaining view, it permutes features among the rows selected for that view. Targets stay unchanged, and same-class matches can remain. Do not substitute arbitrary cross-view noise or change seed rounding/aggregation.

## Validation use (baseline and candidates)

Fit models and learned preprocessing only on the designated training observations.
Validation features and labels are for evaluation and method, hyperparameter or
checkpoint selection only. This restriction applies equally to the baseline and
every candidate, including final retraining. Do not merge validation into training,
fit normalization/encoders on validation, distill from validation predictions, or
relabel validation observations as training or calibration data.

For tasks defined as per-instance inference, optimization or online interaction,
use only the observation/training/calibration/feedback roles explicitly specified
for that instance. Do not use its evaluation targets or carry fitted state from
validation cases into other independent cases. A development replay directory may
contain separately designated training observations; its validation observations
remain excluded from fitting. Public visibility does not authorize training use.

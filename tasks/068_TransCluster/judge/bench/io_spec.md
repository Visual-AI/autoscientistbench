# Clustering tensors

`train.sh --out DIR` learns three independently seeded reusable models. Online Normal batches follow `data/train/generator.json`; scheduler validation is independently generated before training. `models.json` lists the three final checkpoints. The prediction adapter exports `load_model(model_dir,index)` and `step(model,samples,centers)`. One call performs one clustering update; the frozen runner makes exactly twenty calls from the supplied initial centers. The transformer architecture and surrogate loss may change within the 21,672-real-parameter budget. The update must be computed by a learned transformer built from attention with residual connections, with optional feed-forward blocks. Transformer variants are allowed; a hand-written or analytic rule such as Lloyd's algorithm is not. Generic fitted states require `model0.parameters.npz` through `model2.parameters.npz` containing all learned numeric arrays; the runner counts entries, twice for complex values. Torch model parameters are counted directly. Optional `parameter_count(model)` must agree. Fitted parameters hidden as constants violate this contract. Within one `step` call, each fitted layer runs at most once, so a call cannot apply a layer repeatedly to iterate. Hand-written clustering computations, such as hard assignments, Lloyd means, re-seeding or farthest-point selection, may not run before, after or around the transformer inside `step`, and their results may not be given to it as input features.

Input NPZ, no pickle: `id` is an ordered unique opaque string vector, `x` is float32 `[N,512,32]`, `initial` is float32 `[N,10,32]`. Each coordinate was scaled to [0,1] within its task. Initial centers are the first ten randomly permuted points. No family, generative seed or latent parameter appears in inputs.

Prediction NPZ (write through a file handle so the supplied filename is exact): keys exactly `id`, `centers`, `parameters`, `inference_s`. `id` equals the input in order; `centers` is float32 `[3,N,21,10,32]`, including the supplied initial centers at slot zero and the final centers at slot 20. Every slot records the actual corresponding update; repeating an unbounded solver result is not a valid twenty-update experiment. `parameters` is an integer length-three vector; `inference_s` is a nonnegative finite length-three vector. All arrays are finite. Parameter fields are binding claims checked by the scorer and replay; elapsed times are diagnostics.

Labels contain observations, initial centers, distribution families, generation metadata, the baseline parameter budget and OOD thresholds. There are no target cluster assignments: the target is the k-means objective recomputed from observed points and submitted centers. The scorer uses torch float32 Euclidean distances, sums squared minimum distances over points, logs per task, averages within family and model, then takes the median across three models.

The frozen runner loads fresh fitted state for each batch and gives each of three members an independent algorithm stream. Predictions may use only each problem's observations and permitted fitted state; independent problems cannot supply shared inference data. Honor BENCH_SEED and preserve the twenty-update horizon and capacity budget. Internal iterative refinements cannot evade the update count.

`check_model.py MODEL --code CODE --input INPUT --pred PRED --baseline-timing TIMING --timing-input NORMAL_INPUT --seed S --cpus C` replays in the trusted environment. Every final per-task log objective must match within absolute tolerance 1e-4. This verifies the final scored result and allows center-label permutations. Intermediate slots record the actual update sequence; shape, initial state, capacity and replay are checked. The baseline-timing flag is optional and unused. The optional timing-input flag supports diagnostic measurements. The evaluator must run this probe before accepting a score. Input provenance and absence of cross-instance fitting require patch audit.

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

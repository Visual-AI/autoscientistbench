# Private STAR FFT protocol

The outer input is a JSON request with `task=icml-3290-LTARC` and
`protocol=private-policy-v2`. `suite=star-original-100` denotes the formal 4218-row,
100-repetition experiment with 18 covariate columns and tau=.35.
`suite=ist-public-development-v1` denotes the public development set: 18298 real
International Stroke Trial patients, 53 covariate columns,
40 repetitions and tau=.225. It shares no rows with STAR. Its score is a guide to
whether a method change helps and not an estimate of the hidden STAR score.

The development columns encode the paper's 23 covariates. RDELAY, AGE, RSBP and
HOURLOCAL are numeric and unscaled. SEX (True is female), RSLEEP, RATRIAL, RCT, RVISINF, RHEP24
and RASP3 are Boolean. RCONSC, RDEF1 to RDEF8, STYPE, DAYLOCAL and RXHEP give one
Boolean column per value, named `NAME_VALUE`. Treatment 1 is aspirin (RXASP=Y) and
loss 1 is death at six-month follow-up (FDEAD=Y); heparin stays a covariate.

The outer output is an inert stored ZIP archive of candidate code and model
settings, exported by the private-policy-v2 runtime. It is not a table
of decisions. Output filenames may have any extension. The validator checks the
archive; the trusted scorer privately fits and certifies each repetition.

For each phase, `policy.sh --model DIR --input DIR --output FILE` receives a fresh
filesystem/network/process sandbox and a read-only input directory. `request.json`
contains `phase`, `names`, `boolean`, `tau`, `n_bins=40`, `max_splits=3`,
`gamma=1` and `propensity=.5`. `names`, `boolean` and `tau` depend on the suite, so
read them from the request. `data.npz` contains only `x`, `a`, `y` for the
currently authorized role. Rows have no original IDs. All arrays are loaded with
`allow_pickle=False`. Resources are empty, including in formal replay.

During `phase=fit`, only learning rows are available. Return JSON of the form:

```json
{"candidates":[{"rules":[{"criterion":0,"decision":1}],"default":0}]}
```

Each tree is an ordered fast-and-frugal decision list: at most three rules, binary
exit decisions, and one binary default leaf. `criterion` is a nonnegative integer
index into the original training-derived condition grid. Numerical conditions are
ordered by feature, then unique descending percentiles at
`np.linspace(0,100,41,endpoint=False)[1:]`, excluding the training minimum/maximum;
each threshold contributes its original `[True,False]` and `[False,True]` pair.
Boolean conditions contribute the original two labels at value True. The scorer
rebuilds this grid from learning covariates, so arbitrary feature combinations,
thresholds or extra tree attributes cannot be submitted. No more than 100000
candidates or 64 MiB of certificate JSON is accepted as an output safety limit.

The original traversal exits when the condition's continuation mask is false.
Numerical masks use `pandas.cut` with training min/max, `include_lowest=True` and
the original labels. Out-of-range NaNs convert to True as in the source. Boolean
conditions retain the source's equality mask even for its reverse label. These
edge conventions are part of the frozen reconstruction.

During `phase=certify`, only certification rows are available. `request.json`
also supplies the trusted `criteria` and the already validated `candidates` list.
Return exactly `{"selected": INDEX}`. The scorer accepts only an existing integer
index, then reconstructs that immutable tree on evaluation covariates itself.
There is no candidate prediction stage with evaluation rows. Certification method
changes must be statistically valid and may only select precommitted trees; the
interface cannot serve as permission to refit on certification outcomes. Patch
audit evaluates certification validity beyond the structural and access checks.

All 100 formal partitions retain seed 10603+i and original 50/25/25 allocation.
The 40 development partitions use seed 10603+i, i=0,...,39, on the same allocation
rule: 9149 learning, 4574 certification and 4575 evaluation rows.
Each phase and repetition starts fresh; no private files, outputs or mutable state
are made available to another worker or to the agent. Only aggregate statistics
leave the scorer. Failures return a fixed reason, never candidate stderr/logs.

For each repetition, population risk is mean `2*y*I(decision=a)`. Treatment risk is
the same estimated loss averaged over treated decisions, with the released
exceptions: no treatment -> 0, positive treatment with zero estimated loss -> 1.
The 100 population and treatment risks are averaged equally. Coverage is the
fraction of repetitions whose treatment risk <= .35. Both coverage >= .90 and
mean treatment risk <= .35 must hold. Statistical failure yields **1.0**;
structural or execution failure yields `Infinity`. The development set applies the
same rules over its 40 repetitions with tolerance .225 in place of .35.

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

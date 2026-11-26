# Subject encoder source

Sanitized official source from `memory-formation/platonic-representations-fmri`, upstream commit `e28c35a00656354b5e34ec8b85223c508b1c11ef`, with reproduction CPU support and benchmark I/O adaptation.

Use `train.sh` and `predict.sh`. `main.py` implements the offline protocol. The two retained files in `scripts/` contain the linear fitting pipeline and the MLP training loop. Model and estimator definitions are under `src/fmri_mapping/embedding/`. `env.lock` is the runtime dependency source of truth; importing via PYTHONPATH does not install optional dependencies in the upstream project metadata.

See the task README and frozen I/O specification for rules and formats.

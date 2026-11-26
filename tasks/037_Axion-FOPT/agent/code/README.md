# Trainable xi_model adaptation

The safe local base contains selected original source modules and tests without
upstream Git history, target tables, released global fits or predictions.

Run train.sh to fit from the provided training rows. fit.py implements a new
bounded Powell solver; baseline.py defines neutral initialization, fitting bounds
and the editable Eq.25 expression. predict.py exports an inert expression
certificate and numeric predictions. Frozen independent calibration and the
trusted expression evaluator live in the benchmark interface.

The original author global fitting optimizer was not supplied. This adaptation
does not claim to recover it. Formal fit quality and runtime remain unmeasured.

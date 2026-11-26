"""Fixed complete-population MAE with RMSE as a separate diagnostic."""
import csv
import math


def read_table(path, column):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', column]:
            raise ValueError('incorrect CSV header')
        rows = {}
        for row in reader:
            if None in row or not row['id'] or row['id'] in rows:
                raise ValueError('malformed row or duplicate id')
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError('nonfinite value')
            rows[row['id']] = value
        if not rows:
            raise ValueError('empty table')
    return rows


def score_csv(pred_path, labels_path):
    pred, truth = read_table(pred_path, 'pred'), read_table(labels_path, 'y')
    if list(pred) != list(truth):
        raise ValueError('all required ids must appear once in input order')
    errors = [pred[i] - truth[i] for i in truth]
    mae = math.fsum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise ValueError('nonfinite metric')
    return {'mae': mae, 'rmse': rmse, 'coverage_ok': True, 'n': len(errors)}

#!/usr/bin/env python3
"""Metric exactly as the paper computes it. Usage: scorer.py <pred_file> <labels_file>  -> prints {"<metric>": value}."""
import csv
import json
import sys


def read_csv(path, col):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row[col])
    return out


def main(pred_path, labels_path):
    pred = read_csv(pred_path, "pred")
    lab = read_csv(labels_path, "y")
    ids = sorted(lab)
    se = [(pred[i] - lab[i]) ** 2 for i in ids]
    print(json.dumps({"mse": sum(se) / len(se)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

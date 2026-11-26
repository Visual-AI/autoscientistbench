#!/usr/bin/env python3
"""Materialize observations: four independent N(0,4^2) initial positions per run.
The integer seed controls generation only; output contains no seed or estimand.
"""
import csv
import hashlib
import random
import sys

def generate(path, n, seed, namespace):
    rng = random.Random(seed)
    with open(path, "w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["case_id", "initial_1", "initial_2", "initial_3", "initial_4"])
        for i in range(n):
            key = hashlib.sha256(f"{namespace}:{i}".encode()).hexdigest()[:20]
            out.writerow([key] + [format(rng.gauss(0, 4), ".17g") for _ in range(4)])

if __name__ == "__main__":
    generate(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

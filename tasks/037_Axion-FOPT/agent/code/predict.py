import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.environ["BENCH_BENCH_DIR"])
from expressions import FORMAT, Model, load, read_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cert = load(Path(args.model) / "model.json")
    model = Model(cert)
    rows = read_table(args.input)
    result = dict(format=FORMAT, model=cert,
                  rows=[dict(id=row["id"], xi=model.predict(row)["xi"]) for row in rows])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep a numeric second line so the generic text-corruption checks also
    # exercise this JSON format. Whitespace does not change the certificate.
    first, *rest = result["rows"]
    text = '{"rows":[{"xi":\n' + json.dumps(first["xi"], allow_nan=False)
    text += '\n,"id":' + json.dumps(first["id"]) + '}'
    text += ''.join(',' + json.dumps(row, allow_nan=False) for row in rest)
    text += '],"format":' + json.dumps(FORMAT) + ',"model":' + json.dumps(cert, allow_nan=False) + '}\n'
    target.write_text(text)

if __name__ == "__main__":
    main()

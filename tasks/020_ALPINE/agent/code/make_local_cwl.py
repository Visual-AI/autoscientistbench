#!/usr/bin/env python3
"""Create a local (Docker-free) copy of the authors' CWL tree.

Two purely mechanical, semantically-neutral rewrites are applied so the workflow
can run on a machine without Docker (the authors ship DockerRequirement entries
that point at images which are not published in any public registry):

1. every `python /opt/<script>.py` baseCommand is pointed at the equivalent file
   inside the cloned repository (code/DockerFile/...);
2. `DockerRequirement` entries are moved from `requirements` to `hints`
   (biologically identical semantics; cwltool only accepts --no-container when
   Docker is a hint).

The repository under code/ is left untouched.
"""
import os, shutil, sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "ALPINE", "CWL")
DST = sys.argv[1]

MAP = {
    "/opt/filter_fastq_gz.py": "DockerFile/filter/filter_fastq_gz.py",
    "/opt/classify_read.py": "DockerFile/classify/classify_read.py",
    "/opt/count_classification_table.py": "DockerFile/count/count_classification_table.py",
    "/opt/merge_count_table.py": "DockerFile/merge_table/merge_count_table.py",
}


def rewrite_paths(obj):
    if isinstance(obj, str):
        for container_path, repo_rel in MAP.items():
            obj = obj.replace(container_path, os.path.join(ROOT, "ALPINE", repo_rel))
        return obj
    if isinstance(obj, list):
        return [rewrite_paths(x) for x in obj]
    if isinstance(obj, dict):
        return {k: rewrite_paths(v) for k, v in obj.items()}
    return obj


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    n = 0
    for dirpath, _dirs, files in os.walk(DST):
        for fn in files:
            if not fn.endswith(".cwl"):
                continue
            p = os.path.join(dirpath, fn)
            doc = yaml.safe_load(open(p))
            doc = rewrite_paths(doc)
            # Explicit sibling import paths are needed under PYTHONSAFEPATH.
            command = doc.get("baseCommand", [])
            if isinstance(command, list) and len(command)>1 and command[0]=="python":
                doc.setdefault("requirements", []).append({"class":"EnvVarRequirement","envDef":[{"envName":"PYTHONPATH","envValue":os.path.dirname(command[1])}]})
            for requirement in doc.get("requirements", []):
                if isinstance(requirement,dict) and requirement.get("class")=="ResourceRequirement":
                    requirement["coresMin"]=min(int(requirement.get("coresMin",1)),int(os.environ.get("BENCH_CPUS","1")))
            reqs = doc.get("requirements") or []
            docks = [r for r in reqs if isinstance(r, dict) and r.get("class") == "DockerRequirement"]
            if docks:
                doc["requirements"] = [r for r in reqs if r not in docks]
                if not doc["requirements"]:
                    del doc["requirements"]
                hints = doc.get("hints") or []
                doc["hints"] = hints + docks
                n += 1
            with open(p, "w") as fh:
                yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, width=1000)
    print(f"[cwl] wrote {DST} ({n} files: DockerRequirement -> hints, container paths -> repo)")


if __name__ == "__main__":
    main()

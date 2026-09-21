#!/usr/bin/env python3
"""Fetch optional vendored dependencies that are not shipped in this repository.

GROVER supplies pretrained molecular embeddings used when regenerating
``data/drug_data/smiles.grover``. cooclassifier supplies the DLBCL marker panel
used by the patient-stratification analyses.

Usage:
    python scripts/download_grover.py              # both
    python scripts/download_grover.py grover
    python scripts/download_grover.py cooclassifier

Local mirrors (optional): set ``PAIRWISE_VENDOR_MIRROR`` to a directory that
contains ``grover_main/`` and/or ``cooclassifier-main/``; those are preferred
over a network download.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(ROOT, "vendor")

SOURCES = {
    "grover": {
        "local_names": ["grover_main", "grover"],
        "url": "https://github.com/tencent-ailab/grover/archive/refs/heads/main.zip",
        "dest": os.path.join(VENDOR, "grover"),
        "note": "Tencent GROVER molecular graph transformer",
    },
    "cooclassifier": {
        "local_names": ["cooclassifier-main", "cooclassifier"],
        "url": None,
        "dest": os.path.join(VENDOR, "cooclassifier"),
        "note": "LPS / mSHAP DLBCL classifier (R package)",
    },
}


def _candidate_locals(names):
    mirror = os.environ.get("PAIRWISE_VENDOR_MIRROR", "")
    roots = [ROOT, os.path.dirname(ROOT)]
    if mirror:
        roots.insert(0, mirror)
    for root in roots:
        for name in names:
            for sub in ("", "survival_analysis", "pairwise"):
                path = os.path.join(root, sub, name) if sub else os.path.join(root, name)
                if os.path.isdir(path):
                    yield path


def _copytree(src, dest):
    if os.path.exists(dest):
        print(f"  already present: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".git", "wandb"))
    print(f"  copied {src} -> {dest}")


def _download_zip(url, dest):
    os.makedirs(VENDOR, exist_ok=True)
    zip_path = os.path.join(VENDOR, "download.zip")
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(VENDOR)
    extracted = os.path.join(VENDOR, top)
    if os.path.abspath(extracted) != os.path.abspath(dest):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(extracted, dest)
    os.remove(zip_path)
    print(f"  extracted -> {dest}")


def fetch(name):
    spec = SOURCES[name]
    print(f"[{name}] {spec['note']}")
    for local in _candidate_locals(spec["local_names"]):
        _copytree(local, spec["dest"])
        return
    if spec["url"]:
        _download_zip(spec["url"], spec["dest"])
        return
    print(
        f"  no local copy found and no public URL configured.\n"
        f"  set PAIRWISE_VENDOR_MIRROR or place the package at {spec['dest']}.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packages",
        nargs="*",
        choices=sorted(SOURCES),
        default=sorted(SOURCES),
        help="which packages to fetch (default: all)",
    )
    args = parser.parse_args()
    for name in args.packages:
        fetch(name)
    print("done.")


if __name__ == "__main__":
    main()

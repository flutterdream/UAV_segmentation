# -*- coding: utf-8 -*-
"""Create or verify SHA-256 checksums for this portable handoff folder."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST_SHA256.tsv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_manifest() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Manifest exists; will not overwrite: {MANIFEST}")
    paths = sorted((path for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST),
                   key=lambda path: path.relative_to(ROOT).as_posix())
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as output:
        output.write("sha256\tbytes\trelative_path\n")
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(ROOT).as_posix()
            output.write(f"{digest(path)}\t{path.stat().st_size}\t{relative}\n")
            if index % 1000 == 0:
                print(f"hashed {index}/{len(paths)}", flush=True)
    print(f"Manifest complete: {len(paths)} files", flush=True)


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "sha256\tbytes\trelative_path":
        raise RuntimeError("invalid manifest header")
    expected_paths = set()
    for index, line in enumerate(lines[1:], 1):
        expected, size, relative = line.split("\t", 2)
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != expected:
            raise RuntimeError(f"missing, modified or corrupted: {relative}")
        expected_paths.add(relative)
        if index % 1000 == 0:
            print(f"verified {index}/{len(lines) - 1}", flush=True)
    actual_paths = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
                    if path.is_file() and path != MANIFEST}
    extra = actual_paths - expected_paths
    if extra:
        raise RuntimeError(f"unexpected files: {sorted(extra)[:5]}")
    print(f"Verified: {len(expected_paths)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    write_manifest() if args.write_manifest else verify_manifest()

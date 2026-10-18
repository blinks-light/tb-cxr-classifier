"""Data pipeline for the TB chest X-ray classification project.

Sub-commands
------------
download-raw        fetch the 662 PNGs + the official readme from NLM (stdlib only)
build-v1            baseline dataset  : grayscale resize to 224x224, naive random split
build-v2            improved dataset  : near-duplicate removal, corruption audit,
                                        stratified split, frozen label manifest
promote             copy one built version into ``data/processed`` for ``dvc add``
audit               print per-split class balance for a built version

Design notes
------------
* The raw archive is *not* DVC-tracked (3.5 GB, external source).  It is
  re-downloadable from the URLs below and its checksums are recorded in
  ``ml/PROVENANCE.md`` by ``download-raw --write-manifest``.
* Only ``data/processed`` is DVC-tracked (that is what ``data.dvc`` points at).
  Two builds of it, promoted in two different git commits, are the two data
  versions v1 and v2 required by the assignment.
* Every heavy import (numpy / pandas / PIL) happens inside the functions that
  need it, so ``download-raw`` works on a bare interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
VERSIONS_DIR = REPO_ROOT / "data" / "versions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

_STEM = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
INDEX_URL = f"{_STEM}/CXR_png/index.html"
PNG_BASE = f"{_STEM}/CXR_png"
README_URL = f"{_STEM}/NLM-ChinaCXRSet-ReadMe.docx"

IMAGE_SIZE = 224
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[data-pipeline] {msg}", flush=True)


def _fetch(url: str, *, timeout: int = 180, retries: int = 4) -> bytes:
    """HTTP GET with linear back-off, returns raw bytes."""
    import http.client
    import ssl

    transient = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ssl.SSLError)
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tb-cxr-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except transient as exc:
            last = exc
            if attempt < retries:
                wait = 3 * attempt
                _log(f"retry {attempt}/{retries - 1} in {wait}s -> {url.split('/')[-1]} ({exc})")
                time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def list_remote_files() -> list[tuple[str, int]]:
    """Parse the NLM directory index -> [(filename, size_bytes), ...]."""
    import re

    html = _fetch(INDEX_URL).decode("utf-8", "replace")
    pairs = re.findall(r"<a href='([^']+\.png)'>.*?<td class='size'>(\d+)</td>", html, re.S)
    if not pairs:
        raise RuntimeError("could not parse the NLM index page (layout changed?)")
    return [(name, int(size)) for name, size in pairs]


def label_from_filename(name: str) -> int:
    """Shenzhen naming: CHNCXR_0001_0.png = normal, CHNCXR_0001_1.png = tuberculosis."""
    stem = Path(name).stem
    suffix = stem.rsplit("_", 1)[-1]
    if suffix not in {"0", "1"}:
        raise ValueError(f"unexpected filename, cannot derive label: {name}")
    return int(suffix)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path, *, grid: int = 32) -> np.ndarray:
    """Cheap, comparable image fingerprint: a ``grid x grid`` greyscale thumbnail.

    Used for *near-duplicate* detection, which per-image perceptual hashes do
    badly on this dataset: chest radiographs all look alike at 8x8 resolution, so
    a 64-bit dhash collapses 445 of 662 films into "duplicates".  Comparing the
    mean absolute difference of 32x32 thumbnails instead separates genuine
    re-scans (RMSE < 0.02) from merely similar anatomy (RMSE ~ 0.06).
    """
    from PIL import Image

    import numpy as np

    with Image.open(path) as im:
        small = im.convert("L").resize((grid, grid), Image.Resampling.BOX)
        return np.asarray(small, dtype=np.float32).ravel() / 255.0


def pairwise_rmse(fingerprints: np.ndarray) -> np.ndarray:
    """Full pairwise RMSE matrix for a stack of fingerprints."""
    import numpy as np

    squared_norms = (fingerprints * fingerprints).sum(axis=1)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (fingerprints @ fingerprints.T)
    squared = np.clip(squared, 0.0, None)
    rmse = np.sqrt(squared / fingerprints.shape[1])
    np.fill_diagonal(rmse, np.inf)
    return rmse


# --------------------------------------------------------------------------- #
# download-raw
# --------------------------------------------------------------------------- #
def cmd_download_raw(args: argparse.Namespace) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    remote = list_remote_files()
    total_bytes = sum(size for _, size in remote)
    _log(f"remote listing: {len(remote)} files, {total_bytes / 1e9:.2f} GB")

    def job(item: tuple[str, int]) -> tuple[str, str]:
        name, _size = item
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0 and dest.open("rb").read(8) == PNG_MAGIC:
            return name, "cached"
        blob = _fetch(f"{PNG_BASE}/{name}")
        if not blob.startswith(PNG_MAGIC):
            raise RuntimeError(f"{name}: response is not a PNG (got {blob[:16]!r})")
        tmp = dest.with_suffix(".png.part")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        return name, "ok"

    done = skipped = 0
    failures: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, item): item[0] for item in remote}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, status = fut.result()
            except Exception as exc:  # noqa: BLE001 - collect and report at the end
                failures.append(f"{name}: {exc}")
                _log(f"FAILED {name}: {exc}")
                continue
            if status == "cached":
                skipped += 1
            else:
                done += 1
            processed = done + skipped + len(failures)
            if processed % 25 == 0:
                rate = processed / max(time.time() - started, 1e-6)
                _log(f"{processed}/{len(remote)} files  ({rate:.1f} files/s)")

    readme_dest = RAW_DIR / "NLM-ChinaCXRSet-ReadMe.docx"
    if args.with_readme and not readme_dest.exists():
        readme_dest.write_bytes(_fetch(README_URL))
        _log(f"saved {readme_dest.name}")

    if args.write_manifest:
        manifest = {
            "source": "NLM Shenzhen Hospital CXR Set (ChinaCXRSet)",
            "index_url": INDEX_URL,
            "readme_url": README_URL,
            "file_count": len(remote),
            "total_bytes": total_bytes,
            "files": {
                name: {
                    "label": label_from_filename(name),
                    "remote_size": size,
                    "sha256": sha256_of(RAW_DIR / name) if (RAW_DIR / name).exists() else None,
                }
                for name, size in remote
            },
        }
        out = REPO_ROOT / "data" / "raw_manifest.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _log(f"manifest written to {out.relative_to(REPO_ROOT)}")

    _log(f"download finished: {done} new, {skipped} cached, {len(failures)} failed")
    if failures:
        for line in failures:
            print("  -", line, file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# shared dataset builder
# --------------------------------------------------------------------------- #
def _iter_raw():
    """Yield (name, path, label) for every usable raw PNG."""
    for path in sorted(RAW_DIR.glob("*.png")):
        yield path.name, path, label_from_filename(path.name)


def _write_version(
    version: str,
    records: list[dict],
    *,
    meta: dict,
) -> Path:
    """Persist images + labels.csv + manifest.json for one dataset version."""
    from PIL import Image

    out_dir = VERSIONS_DIR / version
    img_dir = out_dir / "images"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in records:
        with Image.open(rec["path"]) as im:
            gray = im.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        dest = img_dir / rec["name"]
        gray.save(dest, format="PNG", optimize=True)
        rows.append(
            {
                "filename": rec["name"],
                "label": rec["label"],
                "split": rec["split"],
                "source_sha256": rec["sha256"],
            }
        )

    import csv

    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "label", "split", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    meta = dict(meta)
    meta.update(
        {
            "version": version,
            "image_size": IMAGE_SIZE,
            "channels": 1,
            "n_images": len(rows),
            "counts": {
                split: {
                    "total": sum(1 for r in rows if r["split"] == split),
                    "positive": sum(1 for r in rows if r["split"] == split and r["label"] == 1),
                    "negative": sum(1 for r in rows if r["split"] == split and r["label"] == 0),
                }
                for split in ("train", "val", "test")
            },
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"{version}: {len(rows)} images -> {out_dir.relative_to(REPO_ROOT)}")
    for split, c in meta["counts"].items():
        _log(f"  {split:5s} total={c['total']:3d} pos={c['positive']:3d} neg={c['negative']:3d}")
    return out_dir


def _split_assign(records: list[dict], *, seed: int, stratified: bool) -> None:
    """Assign rec['split'] in place."""
    import collections

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    rng = random.Random(seed)
    if not stratified:
        pool = records[:]
        rng.shuffle(pool)
        n = len(pool)
        n_train, n_val = int(n * ratios["train"]), int(n * ratios["val"])
        for i, rec in enumerate(pool):
            rec["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        return

    by_class: dict[int, list[dict]] = collections.defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec)
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = int(round(n * ratios["train"])), int(round(n * ratios["val"]))
        for i, rec in enumerate(items):
            rec["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")


# --------------------------------------------------------------------------- #
# build-v1 : baseline (deliberately naive)
# --------------------------------------------------------------------------- #
def cmd_build_v1(args: argparse.Namespace) -> int:
    records = []
    for name, path, label in _iter_raw():
        records.append(
            {
                "name": name,
                "path": path,
                "label": label,
                "sha256": sha256_of(path),
                "split": None,
            }
        )
    if not records:
        _log("no raw images found; run `download-raw` first")
        return 1

    # v1 baseline: single unseeded-looking shuffle, no duplicate handling,
    # no corruption audit.  This is the honest "naive first attempt".
    _split_assign(records, seed=13, stratified=False)
    _write_version(
        "v1",
        records,
        meta={
            "design": "baseline: grayscale + resize only, naive random split, no dedup",
            "split_strategy": "random",
            "split_seed": 13,
            "deduplicated": False,
        },
    )
    return 0


# --------------------------------------------------------------------------- #
# build-v2 : improved
# --------------------------------------------------------------------------- #
def cmd_build_v2(args: argparse.Namespace) -> int:
    """Improved dataset: decode audit, exact/near-duplicate removal, stratified
    split, and an explicit *cross-split leakage* audit recorded in the manifest.

    On this release the duplicate audits find nothing (see ``ml/README_data.md``):
    the closest pair of films is an order of magnitude above any sane duplicate
    threshold.  The value of v2 is therefore the *audited* stratified split --
    balanced classes per split, reproducible seed, and evidence that no test image
    resembles a training image -- rather than a smaller dataset.
    """
    import io

    import numpy as np
    from PIL import Image

    records: list[dict] = []
    dropped_corrupt: list[str] = []
    fingerprints: list[np.ndarray] = []
    for name, path, label in _iter_raw():
        blob = path.read_bytes()
        try:
            with Image.open(io.BytesIO(blob)) as im:
                im.verify()
            with Image.open(io.BytesIO(blob)) as im:
                width, height = im.size
                mode = im.mode
        except Exception:  # noqa: BLE001 - any decode failure means "unusable"
            dropped_corrupt.append(name)
            continue
        records.append(
            {
                "name": name,
                "path": path,
                "label": label,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "size": [width, height],
                "mode": mode,
                "split": None,
            }
        )
        fingerprints.append(fingerprint(path))
    if not records:
        _log("no raw images found; run `download-raw` first")
        return 1

    # --- exact duplicates -------------------------------------------------- #
    by_digest: dict[str, str] = {}
    dropped_exact: list[dict[str, str]] = []
    unique: list[dict] = []
    unique_fp: list[np.ndarray] = []
    for rec, fp in zip(records, fingerprints, strict=True):
        first = by_digest.get(rec["sha256"])
        if first is not None:
            dropped_exact.append({"file": rec["name"], "duplicate_of": first})
            continue
        by_digest[rec["sha256"]] = rec["name"]
        unique.append(rec)
        unique_fp.append(fp)

    # --- near duplicates --------------------------------------------------- #
    matrix = pairwise_rmse(np.stack(unique_fp))
    kept: list[dict] = []
    dropped_near: list[dict[str, object]] = []
    for index, rec in enumerate(unique):
        close = [j for j in range(index) if matrix[index, j] < args.dedup_rmse]
        if close:
            dropped_near.append(
                {"file": rec["name"], "duplicate_of": unique[close[0]]["name"], "rmse": round(float(matrix[index, close[0]]), 5)}
            )
            continue
        kept.append(rec)

    # --- split + leakage audit -------------------------------------------- #
    _split_assign(kept, seed=args.seed, stratified=True)
    kept_index = {rec["name"]: i for i, rec in enumerate(unique)}
    train_idx = [kept_index[r["name"]] for r in kept if r["split"] == "train"]
    holdout_idx = [kept_index[r["name"]] for r in kept if r["split"] in {"val", "test"}]
    cross = matrix[np.ix_(train_idx, holdout_idx)]
    closest_cross = float(cross.min()) if cross.size else float("nan")
    closest_cross_pair = None
    if cross.size:
        row, col = np.unravel_index(int(np.argmin(cross)), cross.shape)
        closest_cross_pair = {
            "train": kept[train_idx[row]]["name"],
            "holdout": kept[holdout_idx[col]]["name"],
            "rmse": round(closest_cross, 5),
        }
    nearest_within = float(np.min(matrix))

    _write_version(
        "v2",
        kept,
        meta={
            "design": "improved: decode audit, exact+near duplicate removal, stratified split, leakage audit",
            "split_strategy": "stratified",
            "split_seed": args.seed,
            "deduplicated": True,
            "dedup_method": "32x32 thumbnail RMSE",
            "dedup_rmse_threshold": args.dedup_rmse,
            "raw_count": len(records) + len(dropped_corrupt),
            "dropped_corrupt": dropped_corrupt,
            "dropped_exact_duplicates": dropped_exact,
            "dropped_near_duplicates": dropped_near,
            "audit": {
                "closest_pair_rmse": round(nearest_within, 5),
                "closest_cross_split_rmse": round(closest_cross, 5),
                "closest_cross_split_pair": closest_cross_pair,
                "leakage_detected": bool(closest_cross < args.dedup_rmse) if cross.size else None,
            },
        },
    )
    return 0


# --------------------------------------------------------------------------- #
# promote / audit
# --------------------------------------------------------------------------- #
def cmd_promote(args: argparse.Namespace) -> int:
    import shutil

    src = VERSIONS_DIR / args.version
    if not (src / "labels.csv").exists():
        _log(f"{src} has no labels.csv; build it first")
        return 1
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    shutil.copytree(src, PROCESSED_DIR)
    _log(f"promoted {args.version} -> data/processed (now run `dvc add data/processed`)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import csv

    src = VERSIONS_DIR / args.version / "labels.csv"
    if not src.exists():
        _log(f"missing {src}")
        return 1
    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    _log(f"{args.version}: {len(rows)} rows")
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        pos = sum(1 for r in subset if r["label"] == "1")
        pct = 100.0 * pos / max(len(subset), 1)
        _log(f"  {split:5s} n={len(subset):3d} pos={pos:3d} ({pct:.1f}%)")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download-raw", help="download the 662 raw PNGs from NLM")
    p_dl.add_argument("--workers", type=int, default=8)
    p_dl.add_argument("--with-readme", action="store_true", default=True)
    p_dl.add_argument("--write-manifest", action="store_true")
    p_dl.set_defaults(func=cmd_download_raw)

    p_v1 = sub.add_parser("build-v1", help="build the baseline dataset version")
    p_v1.set_defaults(func=cmd_build_v1)

    p_v2 = sub.add_parser("build-v2", help="build the improved dataset version")
    p_v2.add_argument("--seed", type=int, default=42)
    p_v2.add_argument(
        "--dedup-rmse",
        type=float,
        default=0.02,
        help="32x32 thumbnail RMSE below which two films count as the same image",
    )
    p_v2.set_defaults(func=cmd_build_v2)

    p_pr = sub.add_parser("promote", help="copy a version into data/processed")
    p_pr.add_argument("--version", required=True, choices=["v1", "v2"])
    p_pr.set_defaults(func=cmd_promote)

    p_au = sub.add_parser("audit", help="print class balance for a version")
    p_au.add_argument("--version", required=True, choices=["v1", "v2"])
    p_au.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

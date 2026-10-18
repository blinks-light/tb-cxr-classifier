# Data lineage — NLM Shenzhen chest X-ray set

This document answers three questions: **where the data came from**, **how
version v2 differs from v1**, and **which git commit corresponds to which data
version**.

---

## 1. Source

| | |
|---|---|
| Dataset | NLM ChinaCXRSet — Shenzhen Hospital CXR set (Tuberculosis Chest X-ray Datasets) |
| Publisher | U.S. National Library of Medicine, Lister Hill National Center for Biomedical Communications |
| Images | <https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/index.html> |
| Description | <https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/NLM-ChinaCXRSet-ReadMe.docx> |
| Size | 662 frontal chest radiographs, 3,768,676,455 bytes (3.51 GiB) of PNG |
| Labels | 336 tuberculosis (`*_1.png`) / 326 normal (`*_0.png`) |
| ReadMe checksum | `sha256 7bff523f767aa29826bfe2c2f1f52bbf79f7b1b66bba857bd8725ef31c8d6dbe` (52,996 bytes) |
| Licence / terms | as published on the NLM page; cite the dataset in any derived work |

The label is encoded in the filename suffix — `CHNCXR_0001_0.png` is normal,
`CHNCXR_0001_1.png` is tuberculosis — and is parsed by
`ml/data_pipeline.py::label_from_filename`.

The raw archive is **not** DVC-tracked: at 3.5 GiB it belongs in a bucket, not in
a course repository. It is reproducible from the index URL, and every file's
SHA-256 is recorded in [`data/raw_manifest.json`](../data/raw_manifest.json)
(662 entries, 0 missing digests) which *is* committed.

---

## 2. Pipeline

```
NLM index ──download-raw──► data/raw/*.png            (3.5 GiB, git-ignored)
                                   │
                                   ├──build-v1──► data/versions/v1  ──promote──► data/processed ──dvc add──► data.dvc (v1)
                                   │
                                   └──build-v2──► data/versions/v2  ──promote──► data/processed ──dvc add──► data.dvc (v2)
```

```powershell
python ml/data_pipeline.py download-raw --write-manifest   # 662 files, resumable, 24 workers
python ml/data_pipeline.py build-v1                         # baseline dataset
python ml/data_pipeline.py build-v2 --seed 42                # audited dataset
python ml/data_pipeline.py promote --version v2              # copy into data/processed
dvc add data/processed && dvc push                           # pointer in git, bytes in Azure
```

Both builds store **224×224 8-bit greyscale PNGs** plus a `labels.csv`
(`filename,label,split,source_sha256`) and a `manifest.json` describing the build.
`data/versions/` is a local working area (git-ignored); the version of record is
`data/processed`, reachable through `data.dvc` in each commit.

### DVC remote

```ini
# .dvc/config  (committed)
[core]
    remote = myazure
['remote "myazure"']
    url = azure://dvc/tb-cxr
    account_name = wut2026
```

Credentials (`sas_token`) live in `.dvc/config.local`, which is git-ignored and
docker-ignored. `dvc pull` on a fresh clone therefore needs that file (or a
`dvc remote modify --local` call) — documented in the README quick start.

---

## 3. v1 → v2

Both versions contain **the same 662 images**; what changes is the split and the
evidence behind it.

| | **v1 (baseline)** | **v2 (improved)** |
|---|---|---|
| Purpose | honest first attempt | audited, balanced dataset |
| Split strategy | single random shuffle, unstratified | stratified per class |
| Seed | 13 | 42 |
| Decode audit | none | every file opened + `Image.verify()` (0 failures) |
| Exact-duplicate removal | none | by SHA-256 (0 removed) |
| Near-duplicate removal | none | 32×32 thumbnail RMSE < 0.02 (0 removed) |
| Cross-split leakage audit | none | yes (see §4) |
| train | 463 (241 pos = 52.1%) | 463 (235 pos = 50.8%) |
| val | 99 (45 pos = 45.5%) | 99 (50 pos = 50.5%) |
| test | 100 (50 pos = 50.0%) | 100 (51 pos = 51.0%) |
| `labels.csv` SHA-256 | `2c44fdb7e5a7ff6bee56a8a2bbdfeb2eccd82100279cbf761c938d9b66b8d037` | `8697e1278cb0abec6e82c7e1110f27db398c4f9cd4ae70d121e2ebc4f1cb0a28` |
| DVC hash (`data.dvc`) | `0fc29fbfacc122686e9566260246cee0.dir` | `b2a03c53e7ebf76eff5746ecfde077d4.dir` |
| Payload size | 17,212,957 B (664 files) | 17,213,444 B (664 files) |
| Git commit | `9be3a10` | `be783cb` |

312 of the 662 images land in a different split in v2 — the two datasets are
materially different training problems, not a cosmetic re-label.

### Why the v1 split is a problem

An unstratified 15 % validation slice drew 45.5 % positives against 52.1 % in
training. With 99 validation images a 7-point class shift is ~7 patients, which is
enough to make a threshold tuned on that slice unreliable — and threshold choice
is the single biggest lever in this project (§5 of `ml/README_experiments.md`).

---

## 4. Audit results (the interesting part)

**There are no duplicates in this release.** That is a finding, not an assumption:

* exact duplicates (SHA-256): **0**
* near duplicates (32×32 thumbnail RMSE < 0.02): **0**
* closest pair of films in the whole set: `CHNCXR_0021_0.png` ↔ `CHNCXR_0321_0.png`,
  **RMSE 0.0596** — three times above the duplicate threshold, i.e. genuinely
  different anatomy
* closest *cross-split* pair (train vs val/test): `CHNCXR_0179_0.png` ↔
  `CHNCXR_0237_0.png`, **RMSE 0.0603** → `leakage_detected: false`

A negative result is worth recording because it rules out the cheapest
explanation for a good test score. The first attempt at this audit used a 64-bit
dHash with Hamming ≤ 6 and flagged 445 of 662 films as duplicates — perceptual
hashes are useless on a dataset where every image is a chest X-ray. The RMSE
approach on 32×32 thumbnails is what the committed pipeline uses.

Because dedup found nothing, **v2's data-level improvement is the audited
stratified split**. All of the accuracy gain in §`ml/README_experiments.md`
therefore has to come from modelling and preprocessing, which is how the two
documents stay honest with each other.

---

## 5. Reproducing a version

```powershell
git checkout 9be3a10          # or be783cb for v2
dvc checkout                  # restores data/processed for that commit
```

`data.dvc` in that commit pins the exact `.dir` hash listed in §3, and the bytes
come from Azure (`dvc push` was run for **both** versions, so both remain
retrievable from the remote cache).

## 6. Known limitations

1. **No patient identifiers.** The public release does not ship them, so splits
   are image-level. The leakage audit bounds the *visual* risk (no image
   resembles another), but two films of the same patient could still be split
   across train and test.
2. **Single centre.** All films come from Shenzhen Hospital; no external
   validation set exists here.
3. **Labels taken as given.** No radiologist re-reading; `_0`/`_1` is the
   published label.
4. The 3.5 GiB raw archive is re-downloadable but not cached in DVC — if NLM
   ever moves the URL, `data/raw_manifest.json` still records what the bytes
   were.

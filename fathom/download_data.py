"""download_data.py -- fetches the real "Ships in Satellite Imagery"
dataset from the Hugging Face Hub once and caches it to disk at
fathom.data.DATA_CACHE_DIR, so every later run (training, evaluation,
tests, CI) works from the same cached copy with no repeated network
dependency.

Usage:
    python3 -m fathom.download_data
"""

from __future__ import annotations

from fathom.data import DATA_CACHE_DIR

HF_DATASET_ID = "jonathan-roberts1/Ships-In-Satellite-Imagery"


def main() -> int:
    if DATA_CACHE_DIR.exists():
        print(f"Already cached at {DATA_CACHE_DIR} -- delete it first to force a re-download.")
        return 0

    from datasets import load_dataset  # noqa: PLC0415

    print(f"Downloading {HF_DATASET_ID} from the Hugging Face Hub...")
    ds = load_dataset(HF_DATASET_ID, split="train")
    print(f"Downloaded {len(ds)} real images. Caching to {DATA_CACHE_DIR} ...")
    DATA_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_CACHE_DIR))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

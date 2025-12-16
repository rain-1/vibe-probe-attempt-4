"""
Train probes for all polysemous words in the dataset by invoking existing training scripts.

This script will iterate over all words from the dataset and invoke
`scripts/train_polysem_probe.py` and `scripts/train_polysem_contrastive.py` via subprocess.

Usage:
  python scripts/train_all_polysems.py --dataset polysemous_datasets.json --device cuda

Be cautious: this will download models and can take significant time/GPU.
"""
import argparse
import subprocess
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="polysemous_datasets.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="checkpoints/")
    p.add_argument("--model", type=str, default="gpt2")
    return p.parse_args()


def load_dataset_words(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


def main():
    args = parse_args()
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        return

    words = load_dataset_words(ds_path)
    print(f"Found {len(words)} words: {words}")

    for w in words:
        print(f"Training linear probe for: {w}")
        cmd1 = ["python", "scripts/train_polysem_probe.py", "--word", w, "--dataset", str(ds_path), "--device", args.device, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--probes-out", args.output_dir, "--model", args.model]
        subprocess.run(cmd1, check=True)

        print(f"Training contrastive probe for: {w}")
        cmd2 = ["python", "scripts/train_polysem_contrastive.py", "--word", w, "--dataset", str(ds_path), "--device", args.device, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--probes-out", args.output_dir, "--model", args.model]
        subprocess.run(cmd2, check=True)

    print("All probes trained.")


if __name__ == "__main__":
    main()

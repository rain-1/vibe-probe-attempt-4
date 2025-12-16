"""
Convenience runner: run the probe visualizer over "Alice's Adventures in Wonderland".

Usage:
  python scripts/visualize_alice.py [--output visualizations/alice.html] [--probes checkpoints/]

The script will look for a local `alice_in_wonderland.txt` in the repo root or `data/`.
If not found it will attempt to download from Project Gutenberg and save to repo root.
It then invokes `scripts/visualize_probe.py --probes ... --text-file <path>`.
"""
from pathlib import Path
import sys
import subprocess
import argparse
import shutil
import urllib.request
import ssl

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATHS = [
    REPO_ROOT / "alice_in_wonderland small.txt",
    REPO_ROOT / "alice_in_wonderland.txt",
    REPO_ROOT / "data" / "alice_in_wonderland.txt",
    REPO_ROOT / "data" / "alice.txt",
]
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def find_or_download_alice(save_path: Path) -> Path:
    # Check candidates first
    for p in CANDIDATE_PATHS:
        if p.exists():
            print(f"Found Alice text at: {p}")
            return p

    # Try to download
    print("Alice text not found locally — attempting download from Project Gutenberg...")
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(GUTENBERG_URL, context=ctx) as resp:
            data = resp.read()
            text = data.decode("utf-8", errors="replace")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(text, encoding="utf-8")
            print(f"Downloaded Alice and saved to: {save_path}")
            return save_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually place 'alice_in_wonderland.txt' in the repository root or data/ directory and re-run.")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Run probe visualizer on Alice in Wonderland")
    parser.add_argument("--output", type=str, default="visualizations/alice_probe_viz.html", help="Output HTML path")
    parser.add_argument("--probes", type=str, nargs="+", default=["checkpoints/"], help="Probe files or directory")
    parser.add_argument("--model", type=str, default=None, help="Optional model override")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    args, extra = parser.parse_known_args()

    alice_save = REPO_ROOT / "alice_in_wonderland.txt"
    alice_path = find_or_download_alice(alice_save)

    # Build command to invoke visualize_probe.py
    script = REPO_ROOT / "scripts" / "visualize_probe.py"
    if not script.exists():
        print(f"ERROR: visualize_probe.py not found at {script}")
        raise SystemExit(1)

    cmd = [sys.executable, str(script), "--text-file", str(alice_path), "--output", args.output]
    # Add probes
    for p in args.probes:
        cmd.extend(["--probes", p])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.device:
        cmd.extend(["--device", args.device])

    print("Running visualizer with command:")
    print(" ".join(cmd))

    # Run and stream output
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Visualizer exited with code {ret.returncode}")
        raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()

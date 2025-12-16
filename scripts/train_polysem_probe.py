"""
Train a simple linear probe for a polysemous word dataset.

This script loads the polysemous datasets from `polysem/polysemous_loader.py`,
computes an average final hidden-state representation for each example, and
trains a linear classifier to predict the sense label.

Usage:
  python scripts/train_polysem_probe.py --word bank --probes-out checkpoints/

The produced checkpoint contains `model_state_dict`, `input_dim`, and `num_classes`.
"""
from pathlib import Path
import argparse
import random
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_load_datasets():
    try:
        from polysem.polysemous_loader import load_datasets
        return load_datasets
    except Exception:
        # Fallback: import by file path
        import importlib.util
        repo_root = Path(__file__).resolve().parents[1]
        loader_path = repo_root / "polysem" / "polysemous_loader.py"
        spec = importlib.util.spec_from_file_location("polysemous_loader", str(loader_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probe on polysemous dataset")
    parser.add_argument("--word", type=str, required=True, help="Target word to train probe for (e.g., bank)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="polysemous_datasets.json",
        help="Path to polysemous datasets JSON (default: polysemous_datasets.json)",
    )
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use for representations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-token-position",
        action="store_true",
        help="Use the target word token position as the example representation (fallback to mean if position not found)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--probes-out", type=str, default="checkpoints/", help="Directory to save probe checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _word_to_token_index(text: str, word_index: int, tokenizer, max_len: int = 512) -> int | None:
    """Map a word index (space-split) to an approximate token index.

    Returns the token index of the last token for that word, or None if mapping fails.
    """
    words = text.split()
    if word_index is None or word_index < 0 or word_index >= len(words):
        return None
    prefix = " ".join(words[: word_index + 1])
    try:
        encoded = tokenizer(prefix, truncation=True, max_length=max_len, return_tensors="pt")
        token_ids = encoded['input_ids'][0]
        if len(token_ids) == 0:
            return None
        return int(len(token_ids) - 1)
    except Exception:
        return None


def prepare_features(model, tokenizer, examples, device, max_len=256, use_token_position=False):
    """Prepare feature tensor for examples.

    If `use_token_position` is True, `examples` should be list of SenseExample objects
    (which implement `find_word_position`). Otherwise `examples` can be list of texts.
    """
    model.eval()
    feats = []
    with torch.no_grad():
        if use_token_position:
            for ex in examples:
                text = ex.text
                word_idx = ex.find_word_position()
                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]  # (seq, hidden_dim)

                token_idx = _word_to_token_index(text, word_idx, tokenizer, max_len=max_len)
                if token_idx is None or token_idx < 0 or token_idx >= last_hidden.size(0):
                    vec = last_hidden.mean(dim=0)
                else:
                    vec = last_hidden[token_idx]
                feats.append(vec.cpu())
        else:
            for item in examples:
                # Accept either raw text strings or objects with a `text` attribute
                if isinstance(item, str):
                    text = item
                elif hasattr(item, "text"):
                    text = item.text
                else:
                    text = str(item)

                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]  # (seq, hidden_dim)
                vec = last_hidden.mean(dim=0)  # (hidden_dim,)
                feats.append(vec.cpu())

    return torch.stack(feats, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    load_datasets = import_load_datasets()
    try:
        datasets = load_datasets(args.dataset)
    except FileNotFoundError:
        # Provide helpful guidance about where we looked
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path(args.dataset),
            repo_root / args.dataset,
            repo_root / "data" / args.dataset,
            repo_root / "polysem" / args.dataset,
        ]
        print(f"ERROR: dataset file not found. Tried these locations:")
        for c in candidates:
            print(f"  - {c}")
        print("\nPlease place the dataset JSON at one of the above paths or pass --dataset <path> to the script.")
        return

    if args.word not in datasets:
        print(f"ERROR: word '{args.word}' not found in dataset keys: {list(datasets.keys())}")
        return

    dataset = datasets[args.word]
    train_examples, test_examples = dataset.train_test_split(test_ratio=0.2, seed=args.seed)

    num_classes = len(dataset.senses)
    print(f"Training probe for word '{args.word}' with {num_classes} senses")

    # Load model & tokenizer
    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # Prepare texts and labels
    train_labels = torch.tensor([ex.sense_id for ex in train_examples], dtype=torch.long)
    test_labels = torch.tensor([ex.sense_id for ex in test_examples], dtype=torch.long)

    print("Computing train features...")
    train_feats = prepare_features(model, tokenizer, train_examples, args.device, max_len=args.max_len, use_token_position=args.use_token_position)
    print("Computing test features...")
    test_feats = prepare_features(model, tokenizer, test_examples, args.device, max_len=args.max_len, use_token_position=args.use_token_position)

    input_dim = train_feats.size(1)
    print(f"Feature dim: {input_dim}, train samples: {train_feats.size(0)}, test samples: {test_feats.size(0)}")

    # Move features to device for training
    train_feats = train_feats.to(args.device)
    test_feats = test_feats.to(args.device)
    train_labels = train_labels.to(args.device)
    test_labels = test_labels.to(args.device)

    # DataLoader
    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Probe model
    probe = nn.Linear(input_dim, num_classes).to(args.device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        # Eval
        probe.eval()
        with torch.no_grad():
            logits = probe(test_feats)
            preds = logits.argmax(dim=-1)
            acc = (preds == test_labels).float().mean().item()

        print(f"Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f} — test_acc: {acc:.4f}")

    # Save checkpoint
    out_dir = Path(args.probes_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polysem_{args.word}_probe.pt"
    checkpoint = {
        "model_state_dict": probe.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "word": args.word,
    }
    torch.save(checkpoint, str(out_path))
    print(f"Saved probe to: {out_path}")


if __name__ == "__main__":
    main()

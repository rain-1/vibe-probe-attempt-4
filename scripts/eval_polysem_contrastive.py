"""
Evaluate a contrastive projection by training a linear classifier on projected embeddings.

Saves no files; prints train/test accuracy.
"""
from pathlib import Path
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_load_datasets():
    try:
        from polysem.polysemous_loader import load_datasets
        return load_datasets
    except Exception:
        import importlib.util
        repo_root = Path(__file__).resolve().parents[1]
        loader_path = repo_root / "polysem" / "polysemous_loader.py"
        spec = importlib.util.spec_from_file_location("polysemous_loader", str(loader_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--word", required=True)
    p.add_argument("--dataset", default="polysemous_datasets.json")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--proj", default=None, help="Path to projection checkpoint (W)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-token-position", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args()


def _word_to_token_index(text: str, word_index: int, tokenizer, max_len: int = 512) -> int | None:
    words = text.split()
    if word_index is None or word_index < 0 or word_index >= len(words):
        return None
    prefix = " ".join(words[: word_index + 1])
    try:
        encoded = tokenizer(prefix, truncation=True, max_length=max_len, return_tensors="pt")
        token_ids = encoded["input_ids"][0]
        if len(token_ids) == 0:
            return None
        return int(len(token_ids) - 1)
    except Exception:
        return None


def extract_embeddings(model, tokenizer, examples, device, max_len=256, use_token_position=False):
    model.eval()
    embs = []
    with torch.no_grad():
        if use_token_position:
            for ex in examples:
                text = ex.text
                word_idx = ex.find_word_position()
                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]
                token_idx = _word_to_token_index(text, word_idx, tokenizer, max_len=max_len)
                if token_idx is None or token_idx < 0 or token_idx >= last_hidden.size(0):
                    vec = last_hidden.mean(dim=0)
                else:
                    vec = last_hidden[token_idx]
                embs.append(vec.cpu())
        else:
            for text in examples:
                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]
                vec = last_hidden.mean(dim=0)
                embs.append(vec.cpu())

    return torch.stack(embs, dim=0)


def main():
    args = parse_args()
    load_datasets = import_load_datasets()
    datasets = load_datasets(args.dataset)
    if args.word not in datasets:
        print("Word not in datasets")
        return

    dataset = datasets[args.word]
    train_examples, test_examples = dataset.train_test_split(test_ratio=0.3, seed=42)

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    print("Extracting train embeddings...")
    X_train = extract_embeddings(model, tokenizer, train_examples, args.device, use_token_position=args.use_token_position)
    y_train = torch.tensor([ex.sense_id for ex in train_examples], dtype=torch.long)

    print("Extracting test embeddings...")
    X_test = extract_embeddings(model, tokenizer, test_examples, args.device, use_token_position=args.use_token_position)
    y_test = torch.tensor([ex.sense_id for ex in test_examples], dtype=torch.long)

    # Load projection
    if args.proj is None:
        proj_path = Path("checkpoints") / f"polysem_contrastive_{args.word}_proj.pt"
    else:
        proj_path = Path(args.proj)
    if not proj_path.exists():
        print(f"Projection checkpoint not found: {proj_path}")
        return
    ckpt = torch.load(proj_path, map_location="cpu")
    W = ckpt.get("W")
    if isinstance(W, torch.Tensor):
        W = W
    else:
        W = torch.tensor(W)

    # Project
    Z_train = X_train @ W.T
    Z_test = X_test @ W.T

    # Train linear classifier on projected features
    device = torch.device(args.device)
    Z_train = Z_train.to(device)
    Z_test = Z_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    clf = nn.Linear(Z_train.size(1), int(y_train.max().item()) + 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        clf.train()
        logits = clf(Z_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        train_preds = clf(Z_train).argmax(dim=-1)
        test_preds = clf(Z_test).argmax(dim=-1)
        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

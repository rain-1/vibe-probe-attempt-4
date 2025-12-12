"""
Train probes on code semantic annotations from the code-data-work dataset.

This script:
1. Loads tokenized code data with semantic annotations
2. Runs the code through a model to extract hidden states
3. Trains linear probes to predict annotations from hidden states
4. Supports multiple probe targets: is_keyword, is_string, is_comment, var_scope, var_type, etc.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load .env for WANDB_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Available probe targets in the dataset
BINARY_TARGETS = ["is_number", "is_comment", "is_string", "is_keyword", "is_operator", "is_punctuation", "is_identifier"]
CATEGORICAL_TARGETS = ["var_scope", "var_type"]  # 0-3 values
ALL_TARGETS = BINARY_TARGETS + CATEGORICAL_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(description="Train probes on code semantic annotations")
    parser.add_argument(
        "--data",
        type=str,
        default="code-data-work/tokenized_data/combined.jsonl",
        help="Path to tokenized JSONL data",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=ALL_TARGETS,
        required=True,
        help=f"Target to predict. Binary: {BINARY_TARGETS}, Categorical: {CATEGORICAL_TARGETS}",
    )
    parser.add_argument(
        "--layer",
        type=str,
        nargs="+",
        default=["12"],
        help="Layer(s) to train probes on (default: 12)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-pt",
        help="Model to extract hidden states from (default: google/gemma-3-1b-pt)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max samples to use (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=256,
        help="Max tokens per sample (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/code_{target}_layer_{layer}.pt",
        help="Output path pattern",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="code-probes",
        help="Wandb project name",
    )
    return parser.parse_args()


class LinearProbe(nn.Module):
    """Simple linear probe for binary or multi-class classification."""

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes if num_classes > 2 else 1)
        self.num_classes = num_classes

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def load_data(data_path: str, max_samples: int):
    """Load JSONL data file."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def extract_hidden_states(model, tokenizer, samples: list, layer: int, max_tokens: int, device: str):
    """
    Extract hidden states from model for each sample.
    Returns: (hidden_states, labels_dict) where hidden_states is (total_tokens, hidden_dim)
    """
    model.eval()
    all_hidden = []
    all_labels = {t: [] for t in ALL_TARGETS}

    with torch.no_grad():
        for sample in tqdm(samples, desc="Extracting hidden states"):
            input_ids = sample["input_ids"][:max_tokens]

            # Run through model
            inputs = torch.tensor([input_ids], device=device)
            outputs = model(inputs, output_hidden_states=True, return_dict=True)

            # Get hidden states for specified layer
            hidden = outputs.hidden_states[layer][0].cpu().float()  # (seq_len, hidden_dim)
            all_hidden.append(hidden)

            # Get labels for each target
            for target in ALL_TARGETS:
                if target in sample:
                    labels = sample[target][:max_tokens]
                    # Pad if needed
                    if len(labels) < len(input_ids):
                        labels = labels + [0] * (len(input_ids) - len(labels))
                    all_labels[target].extend(labels[:len(input_ids)])

    # Concatenate all hidden states
    hidden_states = torch.cat(all_hidden, dim=0)

    # Convert labels to tensors
    labels_tensors = {t: torch.tensor(all_labels[t], dtype=torch.long) for t in ALL_TARGETS}

    return hidden_states, labels_tensors


def train_probe(
    X_train, y_train, X_val, y_val,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    use_wandb: bool,
    target_name: str,
    layer: int,
):
    """Train a linear probe."""
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim, num_classes).to(device)

    if num_classes <= 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            if num_classes <= 2:
                loss = criterion(logits, y_batch.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=-1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_preds.extend(preds.cpu().tolist())
            train_targets.extend(y_batch.cpu().tolist())

        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                if num_classes <= 2:
                    loss = criterion(logits, y_batch.float())
                    preds = (torch.sigmoid(logits) > 0.5).long()
                else:
                    loss = criterion(logits, y_batch)
                    preds = logits.argmax(dim=-1)

                val_loss += loss.item() * len(X_batch)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(y_batch.cpu().tolist())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
            })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}")

    # Load best model
    model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            if num_classes <= 2:
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                preds = logits.argmax(dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(y_batch.tolist())

    final_metrics = {
        "accuracy": accuracy_score(val_targets, val_preds),
        "f1": f1_score(val_targets, val_preds, average='macro', zero_division=0),
        "precision": precision_score(val_targets, val_preds, average='macro', zero_division=0),
        "recall": recall_score(val_targets, val_preds, average='macro', zero_division=0),
    }

    return model, best_val_acc, final_metrics


def main():
    args = parse_args()

    # Parse layers
    layers = []
    for l in args.layer:
        if l.lower() == "final":
            layers.append("final")
        else:
            layers.append(int(l))

    print(f"Loading data from: {args.data}")
    print(f"Target: {args.target}")
    print(f"Layers: {layers}")
    print(f"Max samples: {args.max_samples}")

    # Determine if target is binary or categorical
    is_binary = args.target in BINARY_TARGETS
    num_classes = 2 if is_binary else 4  # var_scope and var_type have 4 classes (0-3)

    # Load data
    samples = load_data(args.data, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # Split into train/val by sample (not by token) to avoid leakage
    n_val = int(len(samples) * args.val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    results = {}

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Training probe for layer {layer}")
        print(f"{'='*60}")

        # Extract hidden states
        print("Extracting training hidden states...")
        X_train, labels_train = extract_hidden_states(
            model, tokenizer, train_samples, layer, args.max_tokens_per_sample, args.device
        )
        y_train = labels_train[args.target]

        print("Extracting validation hidden states...")
        X_val, labels_val = extract_hidden_states(
            model, tokenizer, val_samples, layer, args.max_tokens_per_sample, args.device
        )
        y_val = labels_val[args.target]

        print(f"Train tokens: {len(X_train)}, Val tokens: {len(X_val)}")
        print(f"Label distribution (train): {torch.bincount(y_train, minlength=num_classes).tolist()}")
        print(f"Label distribution (val): {torch.bincount(y_val, minlength=num_classes).tolist()}")

        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.target}_layer_{layer}",
                config={
                    "target": args.target,
                    "layer": layer,
                    "model": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "train_tokens": len(X_train),
                    "val_tokens": len(X_val),
                    "num_classes": num_classes,
                },
                reinit=True,
            )

        # Train probe
        probe, best_acc, final_metrics = train_probe(
            X_train, y_train, X_val, y_val,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            use_wandb=use_wandb,
            target_name=args.target,
            layer=layer,
        )

        print(f"\nFinal metrics for layer {layer}:")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  F1:        {final_metrics['f1']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")

        results[layer] = final_metrics

        # Save probe
        output_path = Path(args.output.format(target=args.target, layer=layer))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": probe.state_dict(),
            "layer": layer,
            "target": args.target,
            "num_classes": num_classes,
            "input_dim": X_train.shape[1],
            "best_val_acc": best_acc,
            "final_metrics": final_metrics,
        }, output_path)
        print(f"Saved probe to: {output_path}")

        if use_wandb:
            wandb.finish()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY for target: {args.target}")
    print(f"{'='*60}")
    for layer, metrics in sorted(results.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        print(f"Layer {layer}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    best_layer = max(results, key=lambda l: results[l]['accuracy'])
    print(f"\nBest layer: {best_layer} (Acc={results[best_layer]['accuracy']:.4f})")


if __name__ == "__main__":
    main()

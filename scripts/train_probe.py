"""
Probe training script for linear probes on LLM hidden activations.

Trains a simple linear probe (with optional GELU) to predict the target token
from hidden layer activations, without loading the LLM.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load .env for WANDB_API_KEY
load_dotenv()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear probe on hidden layer activations"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to parquet file with hidden activations",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        required=True,
        help="Which layer(s) to train probes on (e.g., --layer 12 or --layer 0 6 12 18)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Training mode: 'classification' (binary cross-entropy) or 'regression' (MSE on probability)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="L2 regularization / weight decay (default: 1e-2)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--use-gelu",
        action="store_true",
        help="Add GELU activation after linear layer",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vibe-probe",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Weights & Biases run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/probe_layer_{layer}.pt",
        help="Path pattern to save trained probes (default: checkpoints/probe_layer_{layer}.pt)",
    )
    parser.add_argument(
        "--log-every-step",
        action="store_true",
        help="Log loss to wandb every training step (not just every epoch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run to verify dataset integrity (check for train/val overlap) and exit",
    )
    return parser.parse_args()


class LinearProbe(nn.Module):
    """Simple linear probe with optional GELU activation."""
    
    def __init__(self, input_dim: int, use_gelu: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.use_gelu = use_gelu
        if use_gelu:
            self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.linear(x)
        if self.use_gelu:
            x = self.gelu(x)
        return x.squeeze(-1)


def load_data(data_path: str, layer: int, mode: str):
    """Load dataset and extract features/targets for the specified layer."""
    df = pd.read_parquet(data_path)
    
    hidden_col = f"layer-{layer}-hidden"
    if hidden_col not in df.columns:
        available = [c for c in df.columns if c.startswith("layer-") and c.endswith("-hidden")]
        raise ValueError(f"Layer {layer} not found. Available: {available}")
    
    # Extract features
    X = np.array(df[hidden_col].tolist(), dtype=np.float32)
    
    # Extract targets based on mode
    if mode == "classification":
        y = df["label"].values.astype(np.float32)
    else:  # regression
        y = df["target_token_prob"].values.astype(np.float32)
    
    return X, y, df


def compute_metrics(preds, targets, mode):
    """Compute accuracy metrics for validation."""
    if mode == "classification":
        pred_labels = (torch.sigmoid(preds) > 0.5).float()
    else:
        # For regression, use 0.5 threshold on predicted probability
        pred_labels = (torch.sigmoid(preds) > 0.5).float()
    
    # Overall accuracy
    correct = (pred_labels == (targets > 0.5).float()).float()
    accuracy = correct.mean().item()
    
    # Positive accuracy (where target is 1 / high prob)
    pos_mask = targets > 0.5
    pos_acc = correct[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
    
    # Negative accuracy (where target is 0 / low prob)
    neg_mask = targets <= 0.5
    neg_acc = correct[neg_mask].mean().item() if neg_mask.sum() > 0 else 0.0
    
    return accuracy, pos_acc, neg_acc


def train_epoch(model, train_loader, optimizer, criterion, device, mode, log_every_step=False, use_wandb=False, global_step=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    step = global_step
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * len(X_batch)
        all_preds.append(preds.detach())
        all_targets.append(y_batch.detach())
        
        # Log per-step if requested
        if log_every_step and use_wandb:
            wandb.log({"step": step, "step/loss": batch_loss})
        step += 1
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    avg_loss = total_loss / len(all_preds)
    accuracy, pos_acc, neg_acc = compute_metrics(all_preds, all_targets, mode)
    
    return avg_loss, accuracy, pos_acc, neg_acc, step


def validate(model, val_loader, criterion, device, mode):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    avg_loss = total_loss / len(all_preds)
    accuracy, pos_acc, neg_acc = compute_metrics(all_preds, all_targets, mode)
    
    return avg_loss, accuracy, pos_acc, neg_acc

def train_single_layer(args, layer: int, use_wandb: bool):
    """Train a probe for a single layer."""
    print(f"\n{'='*60}")
    print(f"Training probe for layer {layer}")
    print(f"{'='*60}")
    
    # Load data for this layer
    X, y, df = load_data(args.data, layer, args.mode)
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    
    # Train/val split - split indices to track text overlap if needed
    indices = np.arange(len(X))
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, indices, test_size=args.val_split, random_state=42, stratify=(y > 0.5).astype(int)
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Dry run verification
    if args.dry_run:
        print(f"\n{'='*20} DRY RUN VERIFICATION {'='*20}")
        print("Checking for data leakage (overlap between train and validation sets)...")
        
        train_texts = set(df.iloc[idx_train]["text"].tolist())
        val_texts = set(df.iloc[idx_val]["text"].tolist())
        
        intersection = train_texts.intersection(val_texts)
        overlap_count = len(intersection)
        val_total = len(val_texts)
        overlap_ratio = overlap_count / val_total if val_total > 0 else 0
        
        print(f"Unique texts in Train: {len(train_texts)}")
        print(f"Unique texts in Val:   {len(val_texts)}")
        print(f"Overlapping texts:     {overlap_count}")
        print(f"Overlap ratio (of Val): {overlap_ratio:.2%}")
        
        if overlap_count > 0:
            print("\nWARNING: DATA LEAKAGE DETECTED!")
            print("Examples of overlapping texts:")
            for i, text in enumerate(list(intersection)[:5]):
                print(f"{i+1}. {text[:100]}...")
        else:
            print("\nSUCCESS: No overlap detected between training and validation sets.")
            
        print(f"{'='*60}\n")
        return 0.0  # Return dummy accuracy
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    model = LinearProbe(X.shape[1], use_gelu=args.use_gelu).to(args.device)
    
    # Loss function
    if args.mode == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    # Optimizer with weight decay (Ridge regularization)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize wandb for this layer
    if use_wandb:
        run_name = args.wandb_run or f"layer-{layer}-{args.mode}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "layer": layer,
                "mode": args.mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "use_gelu": args.use_gelu,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "input_dim": X.shape[1],
            },
            reinit=True,  # Allow multiple runs in same process
        )
    
    # Training loop
    print("Training...")
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in tqdm(range(args.epochs), desc=f"Layer {layer}"):
        # Train
        train_loss, train_acc, train_pos_acc, train_neg_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion, args.device, args.mode,
            log_every_step=args.log_every_step, use_wandb=use_wandb, global_step=global_step
        )
        
        # Validate
        val_loss, val_acc, val_pos_acc, val_neg_acc = validate(
            model, val_loader, criterion, args.device, args.mode
        )
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/pos_accuracy": train_pos_acc,
                "train/neg_accuracy": train_neg_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/pos_accuracy": val_pos_acc,
                "val/neg_accuracy": val_neg_acc,
            })
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} (P:{train_pos_acc:.3f}, N:{train_neg_acc:.3f}) | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} (P:{val_pos_acc:.3f}, N:{val_neg_acc:.3f})"
            )
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save model with layer in filename
    output_path = Path(args.output.format(layer=layer))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "layer": layer,
        "mode": args.mode,
        "input_dim": X.shape[1],
        "use_gelu": args.use_gelu,
        "best_val_acc": best_val_acc,
    }, output_path)
    print(f"Model saved to: {output_path}")
    
    if use_wandb:
        wandb.finish()
    
    return best_val_acc


def main():
    args = parse_args()
    
    print(f"Loading data from: {args.data}")
    print(f"Mode: {args.mode}")
    print(f"Layers to train: {args.layer}")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    
    # Train probe for each layer
    results = {}
    for layer in args.layer:
        best_acc = train_single_layer(args, layer, use_wandb)
        results[layer] = best_acc
    
    # Summary
    print(f"\n{'='*60}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*60}")
    for layer, acc in sorted(results.items()):
        print(f"Layer {layer:2d}: {acc:.4f}")
    
    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer} ({results[best_layer]:.4f})")


if __name__ == "__main__":
    main()

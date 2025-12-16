"""
Line Length Contrastive Probes

Based on insights from:
- Transformer Circuits paper on linebreaking: https://transformer-circuits.pub/2025/linebreaks/index.html
- Our polysem experiments showing contrastive learning helps

Key changes from previous attempts:
1. Binary task: "line length <= n" vs "line length > n" (not exact length)
2. Contrastive learning to find the length direction
3. Probing the residual stream (hidden states)
4. Using a model that actually sees fixed-width text (GPT-2 trained on code)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Tuple
import random


# ============================================================
# Data Generation
# ============================================================

def generate_line(target_length: int, tolerance: int = 2) -> str:
    """Generate a line of approximately target_length characters."""
    words = [
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "and", "but", "or", "nor", "for", "yet", "so",
        "in", "on", "at", "by", "with", "from", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "not", "only", "own", "same", "than", "too",
        "very", "just", "also", "now", "new", "old", "good", "bad", "high",
        "low", "big", "small", "long", "short", "early", "late", "young",
        "little", "much", "many", "great", "right", "left", "next", "last",
        "first", "second", "third", "public", "private", "local", "national",
        "world", "life", "time", "year", "day", "night", "week", "month",
        "home", "house", "room", "door", "window", "floor", "wall", "street",
        "city", "town", "country", "water", "food", "money", "work", "job",
        "book", "paper", "word", "letter", "number", "part", "place", "case",
        "point", "fact", "thing", "man", "woman", "child", "person", "people",
        "name", "hand", "head", "face", "eye", "body", "side", "end", "line",
        "system", "program", "function", "class", "method", "variable", "value",
        "data", "file", "code", "error", "result", "output", "input", "process",
    ]

    line = ""
    while len(line) < target_length - tolerance:
        word = random.choice(words)
        if line:
            candidate = line + " " + word
        else:
            candidate = word

        if len(candidate) <= target_length + tolerance:
            line = candidate
        elif len(line) >= target_length - tolerance:
            break
        else:
            # Try a shorter word
            short_words = [w for w in words if len(w) <= 4]
            word = random.choice(short_words)
            candidate = line + " " + word if line else word
            if len(candidate) <= target_length + tolerance:
                line = candidate
            else:
                break

    return line


def generate_dataset(
    n_samples: int = 500,
    threshold: int = 40,
    min_length: int = 10,
    max_length: int = 80
) -> Tuple[List[str], List[int], List[int]]:
    """
    Generate dataset of lines with binary labels.

    Returns:
        lines: List of text lines
        labels: 0 if len <= threshold, 1 if len > threshold
        lengths: Actual character lengths
    """
    lines = []
    labels = []
    lengths = []

    # Generate balanced dataset
    for _ in range(n_samples // 2):
        # Short line (below threshold)
        target = random.randint(min_length, threshold)
        line = generate_line(target)
        if len(line) <= threshold:  # Verify
            lines.append(line)
            labels.append(0)
            lengths.append(len(line))

    for _ in range(n_samples // 2):
        # Long line (above threshold)
        target = random.randint(threshold + 1, max_length)
        line = generate_line(target)
        if len(line) > threshold:  # Verify
            lines.append(line)
            labels.append(1)
            lengths.append(len(line))

    return lines, labels, lengths


# ============================================================
# Embedding Extraction
# ============================================================

def extract_embeddings(
    model,
    tokenizer,
    lines: List[str],
    layer: int = -1,
    position: str = "last",  # "last", "mean", "first"
    device: str = "cpu"
) -> torch.Tensor:
    """Extract embeddings from the residual stream."""
    model = model.to(device)
    model.eval()

    embeddings = []

    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                output_hidden_states=True
            )

        hidden = outputs.hidden_states[layer][0]  # (seq_len, hidden_dim)

        if position == "last":
            emb = hidden[-1]
        elif position == "mean":
            emb = hidden.mean(dim=0)
        elif position == "first":
            emb = hidden[0]
        else:
            raise ValueError(f"Unknown position: {position}")

        embeddings.append(emb.cpu())

    return torch.stack(embeddings)


# ============================================================
# Contrastive Learning
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        pos_count = pos_mask.sum(dim=1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=device)

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        self_mask = torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -mean_log_prob[valid].mean()
        return loss


class ContrastiveProbe(nn.Module):
    """Linear projection for contrastive learning."""

    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def train_contrastive_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dim: int = 64,
    epochs: int = 200,
    lr: float = 0.001,
    temperature: float = 0.1,
    batch_size: int = 64,
    verbose: bool = True
) -> ContrastiveProbe:
    """Train contrastive probe."""

    input_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]

    probe = ContrastiveProbe(input_dim, output_dim)
    criterion = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        emb_shuffled = embeddings[perm]
        lab_shuffled = labels[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_emb = emb_shuffled[i:i+batch_size]
            batch_lab = lab_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            projected = probe(batch_emb)
            loss = criterion(projected, batch_lab)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        if verbose and epoch % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}: loss = {avg_loss:.4f}")

    return probe


# ============================================================
# Evaluation
# ============================================================

def evaluate_linear(embeddings: torch.Tensor, labels: torch.Tensor, name: str = "") -> float:
    """Cross-validated logistic regression."""
    X = embeddings.numpy()
    y = labels.numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)

    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    return scores.mean()


def evaluate_contrastive(
    probe: ContrastiveProbe,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    name: str = ""
) -> float:
    """Evaluate on projected space."""
    probe.eval()
    with torch.no_grad():
        projected = probe(embeddings)
    return evaluate_linear(projected, labels, name=name)


# ============================================================
# Main Experiment
# ============================================================

def main():
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    print("=" * 70)
    print("LINE LENGTH CONTRASTIVE PROBES")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    print(f"Model: {model_name} ({n_layers} layers)")

    # Test different thresholds
    thresholds = [30, 40, 50, 60]
    results = {}

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"THRESHOLD: {threshold} characters")
        print(f"{'='*60}")

        # Generate data
        print("\nGenerating dataset...")
        lines, labels, lengths = generate_dataset(
            n_samples=600,
            threshold=threshold,
            min_length=10,
            max_length=80
        )
        labels = torch.tensor(labels)

        print(f"  Short (<=  {threshold}): {(labels == 0).sum().item()}")
        print(f"  Long  (> {threshold}): {(labels == 1).sum().item()}")
        print(f"  Length range: {min(lengths)} - {max(lengths)}")

        threshold_results = {}

        # Try different layers
        layers_to_try = [0, n_layers // 2, -1]

        for layer_idx in layers_to_try:
            layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "last_layer"
            print(f"\n--- {layer_name} ---")

            # Extract embeddings (try different positions)
            for position in ["last", "mean"]:
                print(f"\n  Position: {position}")

                embeddings = extract_embeddings(
                    model, tokenizer, lines,
                    layer=layer_idx,
                    position=position
                )

                # Baseline: raw embeddings
                baseline = evaluate_linear(embeddings, labels, name="Raw embeddings")

                # Contrastive probe
                print("  Training contrastive probe...")
                probe = train_contrastive_probe(
                    embeddings, labels,
                    output_dim=32,
                    epochs=200,
                    lr=0.001,
                    verbose=False
                )

                contrastive = evaluate_contrastive(probe, embeddings, labels, name="Contrastive")

                threshold_results[f"{layer_name}_{position}"] = {
                    "baseline": baseline,
                    "contrastive": contrastive
                }

        results[threshold] = threshold_results

        # Visualize best result
        print("\n  Creating visualization...")
        embeddings = extract_embeddings(model, tokenizer, lines, layer=-1, position="last")
        probe = train_contrastive_probe(embeddings, labels, output_dim=32, epochs=200, verbose=False)

        probe.eval()
        with torch.no_grad():
            projected = probe(embeddings)

        pca = PCA(n_components=2)
        proj_2d = pca.fit_transform(projected.numpy())

        plt.figure(figsize=(8, 6))
        short_mask = labels.numpy() == 0
        long_mask = labels.numpy() == 1

        plt.scatter(proj_2d[short_mask, 0], proj_2d[short_mask, 1],
                   c='blue', alpha=0.5, label=f'<= {threshold} chars', s=30)
        plt.scatter(proj_2d[long_mask, 0], proj_2d[long_mask, 1],
                   c='red', alpha=0.5, label=f'> {threshold} chars', s=30)

        plt.legend()
        plt.title(f'Line Length Probe (threshold={threshold})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.tight_layout()
        plt.savefig(f'line_length_threshold_{threshold}.png', dpi=150)
        plt.close()
        print(f"  Saved: line_length_threshold_{threshold}.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Threshold':<12} | {'Layer':<20} | {'Baseline':<10} | {'Contrastive':<12}")
    print("-" * 60)

    for threshold, thresh_results in results.items():
        for config, accs in thresh_results.items():
            print(f"{threshold:<12} | {config:<20} | {accs['baseline']:<10.3f} | {accs['contrastive']:<12.3f}")


if __name__ == "__main__":
    main()

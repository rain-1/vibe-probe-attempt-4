"""
Line Length Probe Validation

Train probes on synthetic data, then validate on held-out real text.
Generate HTML visualization to verify the probes actually work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import random


# ============================================================
# Contrastive Learning (from main script)
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
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
        return -mean_log_prob[valid].mean()


class ContrastiveProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def train_contrastive_probe(embeddings, labels, output_dim=64, epochs=200, lr=0.001, batch_size=64):
    input_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    probe = ContrastiveProbe(input_dim, output_dim)
    criterion = SupervisedContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        emb_shuffled = embeddings[perm]
        lab_shuffled = labels[perm]
        for i in range(0, n_samples, batch_size):
            batch_emb = emb_shuffled[i:i+batch_size]
            batch_lab = lab_shuffled[i:i+batch_size]
            optimizer.zero_grad()
            projected = probe(batch_emb)
            loss = criterion(projected, batch_lab)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()

    return probe


# ============================================================
# Data Generation
# ============================================================

def generate_line(target_length: int, tolerance: int = 2) -> str:
    words = [
        "the", "a", "an", "is", "was", "are", "be", "been", "have", "has",
        "do", "does", "did", "will", "would", "could", "should", "can",
        "and", "but", "or", "for", "in", "on", "at", "by", "with", "from",
        "then", "when", "where", "how", "all", "each", "both", "few", "more",
        "some", "no", "not", "only", "very", "just", "also", "now", "new",
        "good", "high", "big", "small", "long", "old", "great", "little",
        "world", "life", "time", "year", "day", "home", "work", "system",
        "program", "function", "data", "file", "code", "value", "result",
    ]
    line = ""
    while len(line) < target_length - tolerance:
        word = random.choice(words)
        candidate = line + " " + word if line else word
        if len(candidate) <= target_length + tolerance:
            line = candidate
        elif len(line) >= target_length - tolerance:
            break
        else:
            short_words = [w for w in words if len(w) <= 3]
            word = random.choice(short_words)
            candidate = line + " " + word if line else word
            if len(candidate) <= target_length + tolerance:
                line = candidate
            else:
                break
    return line


def generate_training_data(n_samples=500, threshold=40):
    lines, labels, lengths = [], [], []
    for _ in range(n_samples // 2):
        target = random.randint(10, threshold)
        line = generate_line(target)
        if len(line) <= threshold:
            lines.append(line)
            labels.append(0)
            lengths.append(len(line))
    for _ in range(n_samples // 2):
        target = random.randint(threshold + 1, 80)
        line = generate_line(target)
        if len(line) > threshold:
            lines.append(line)
            labels.append(1)
            lengths.append(len(line))
    return lines, labels, lengths


# ============================================================
# Embedding Extraction
# ============================================================

def extract_embeddings(model, tokenizer, lines, layer=-1, device="cpu"):
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
        hidden = outputs.hidden_states[layer][0]
        emb = hidden.mean(dim=0)  # Mean pooling
        embeddings.append(emb.cpu())
    return torch.stack(embeddings)


# ============================================================
# Validation Text
# ============================================================

# Real formatted text with known line widths
VALIDATION_TEXT = """
The quick brown fox jumps over the lazy dog near the old barn.
A short line.
This is a medium length line with some words.
Programming is the art of telling a computer what to do.
Hi there!
The rain in Spain falls mainly on the plain during the autumn months.
Code review is essential for maintaining software quality in teams.
Yes.
Functions should do one thing and do it well according to best practices.
A tiny one.
The model processes each token sequentially from left to right always.
Testing helps catch bugs before they reach production environments today.
No way!
Machine learning models can learn complex patterns from large datasets.
Short text here.
Natural language processing enables computers to understand human speech.
The transformer architecture revolutionized deep learning for sequences.
OK.
Attention mechanisms allow models to focus on relevant parts of input.
This line is exactly fifty characters long here!
A brief note.
The embedding space captures semantic relationships between words nicely.
Probe training uses contrastive learning to find good directions.
Go!
Validation on held-out data reveals whether probes truly generalize well.
"""


def parse_validation_text(text: str) -> List[Tuple[str, int]]:
    """Parse validation text into (line, length) pairs."""
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line:
            lines.append((line, len(line)))
    return lines


# ============================================================
# HTML Visualization
# ============================================================

def create_html_visualization(
    lines_with_predictions: List[Dict],
    threshold: int,
    accuracy: float,
    output_file: str
):
    """Create HTML visualization of probe predictions."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Line Length Probe Validation</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            max-width: 1000px;
            margin: 40px auto;
            padding: 20px;
            background: #1e1e1e;
            color: #d4d4d4;
        }}
        h1 {{ color: #569cd6; text-align: center; }}
        .stats {{
            text-align: center;
            font-size: 1.2em;
            margin: 20px 0;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 8px;
        }}
        .accuracy {{
            font-size: 1.5em;
            font-weight: bold;
            color: {"#4ec9b0" if accuracy > 0.8 else "#f14c4c" if accuracy < 0.6 else "#dcdcaa"};
        }}
        .threshold-info {{
            color: #9cdcfe;
            margin: 10px 0;
        }}
        .line-container {{
            margin: 8px 0;
            padding: 10px;
            background: #2d2d2d;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        .correct {{ border-color: #4ec9b0; }}
        .wrong {{ border-color: #f14c4c; background: #3d2d2d; }}
        .line-text {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .short {{ color: #4ec9b0; }}
        .long {{ color: #ce9178; }}
        .meta {{
            font-size: 0.85em;
            color: #808080;
            margin-top: 5px;
        }}
        .pred {{ color: #dcdcaa; }}
        .actual {{ color: #9cdcfe; }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .ruler {{
            margin: 20px 0;
            padding: 10px;
            background: #252526;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .ruler-text {{
            color: #569cd6;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <h1>Line Length Probe Validation</h1>

    <div class="stats">
        <div class="threshold-info">Threshold: {threshold} characters</div>
        <div class="threshold-info">Task: Predict if line length ≤ {threshold} (short) or > {threshold} (long)</div>
        <div style="margin-top: 15px;">
            Validation Accuracy: <span class="accuracy">{accuracy:.1%}</span>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #4ec9b0;"></div>
            <span>Short (≤ {threshold} chars)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ce9178;"></div>
            <span>Long (> {threshold} chars)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f14c4c;"></div>
            <span>Wrong prediction</span>
        </div>
    </div>

    <div class="ruler">
        <div class="ruler-text">
            {"".join([str(i % 10) for i in range(1, 81)])}
        </div>
        <div class="ruler-text">
            {"".join(["│" if i % 10 == 0 else " " for i in range(1, 81)])}
            <span style="color: #f14c4c;"> ← {threshold} char threshold</span>
        </div>
    </div>
"""

    for item in lines_with_predictions:
        line = item['line']
        length = item['length']
        pred = item['predicted']
        actual = item['actual']
        prob = item['probability']
        correct = pred == actual

        status_class = "correct" if correct else "wrong"
        text_class = "short" if actual == 0 else "long"
        pred_label = "short" if pred == 0 else "long"
        actual_label = "short" if actual == 0 else "long"

        # Create visual length indicator
        bar_length = min(length, 80)
        bar_color = "#4ec9b0" if actual == 0 else "#ce9178"

        html += f"""
    <div class="line-container {status_class}">
        <div class="line-text {text_class}">{line}</div>
        <div class="meta">
            Length: <span class="actual">{length}</span> |
            Actual: <span class="actual">{actual_label}</span> |
            Predicted: <span class="pred">{pred_label}</span> ({prob:.0%}) |
            {"✓" if correct else "✗"}
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {output_file}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("LINE LENGTH PROBE VALIDATION")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Test multiple thresholds
    thresholds = [30, 40, 50]

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"THRESHOLD: {threshold} characters")
        print(f"{'='*60}")

        # Generate training data
        print("\nGenerating training data...")
        train_lines, train_labels, train_lengths = generate_training_data(
            n_samples=600, threshold=threshold
        )
        train_labels = torch.tensor(train_labels)
        print(f"  Training samples: {len(train_lines)}")

        # Extract training embeddings
        print("Extracting training embeddings...")
        train_embeddings = extract_embeddings(model, tokenizer, train_lines, layer=-1)

        # Train contrastive probe
        print("Training contrastive probe...")
        probe = train_contrastive_probe(train_embeddings, train_labels, output_dim=32, epochs=200)

        # Project training data and fit classifier
        probe.eval()
        with torch.no_grad():
            train_projected = probe(train_embeddings)

        scaler = StandardScaler()
        train_proj_scaled = scaler.fit_transform(train_projected.numpy())

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(train_proj_scaled, train_labels.numpy())

        # Prepare validation data
        print("\nPreparing validation data...")
        val_data = parse_validation_text(VALIDATION_TEXT)
        val_lines = [line for line, _ in val_data]
        val_lengths = [length for _, length in val_data]
        val_labels = [0 if length <= threshold else 1 for length in val_lengths]

        print(f"  Validation samples: {len(val_lines)}")
        print(f"  Short lines: {sum(1 for l in val_labels if l == 0)}")
        print(f"  Long lines: {sum(1 for l in val_labels if l == 1)}")

        # Extract validation embeddings
        print("Extracting validation embeddings...")
        val_embeddings = extract_embeddings(model, tokenizer, val_lines, layer=-1)

        # Project and predict
        with torch.no_grad():
            val_projected = probe(val_embeddings)

        val_proj_scaled = scaler.transform(val_projected.numpy())
        predictions = classifier.predict(val_proj_scaled)
        probabilities = classifier.predict_proba(val_proj_scaled)

        # Calculate accuracy
        correct = sum(p == a for p, a in zip(predictions, val_labels))
        accuracy = correct / len(val_labels)
        print(f"\nValidation Accuracy: {accuracy:.1%} ({correct}/{len(val_labels)})")

        # Prepare data for visualization
        lines_with_predictions = []
        for i, (line, length) in enumerate(val_data):
            lines_with_predictions.append({
                'line': line,
                'length': length,
                'predicted': predictions[i],
                'actual': val_labels[i],
                'probability': probabilities[i][predictions[i]]
            })

        # Create HTML visualization
        create_html_visualization(
            lines_with_predictions,
            threshold,
            accuracy,
            f"line_length_validation_{threshold}.html"
        )

    print("\n" + "=" * 70)
    print("DONE! Open the HTML files to inspect predictions.")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Line Length Probes with Real Fixed-Width Text

The key insight from Transformer Circuits: models learn to track line position
because they predict newlines in fixed-width formatted text during pretraining.

This experiment:
1. Uses actual fixed-width formatted text (code, RFCs, etc.)
2. Tests with both GPT-2 and Qwen
3. Validates on held-out text from the same distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import random
import requests

# ============================================================
# Contrastive Learning
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
# Fixed-Width Text Sources
# ============================================================

# Classic fixed-width formatted text samples
# These mimic what GPT-2 would have seen during pretraining

FIXED_WIDTH_SAMPLES = """
================================================================================
                           SYSTEM INFORMATION REPORT
================================================================================

This document contains technical specifications for the computing system.
All lines in this document are formatted to exactly 80 characters wide.
This is standard practice for terminal displays and text file formatting.

--------------------------------------------------------------------------------
SECTION 1: Hardware Configuration
--------------------------------------------------------------------------------

Processor:      Intel Core i7-9700K @ 3.60GHz
Memory:         32768 MB DDR4-3200
Storage:        1024 GB NVMe SSD
Graphics:       NVIDIA GeForce RTX 2080

The system has been configured for optimal performance in development tasks.
All components have been tested and verified to meet specifications exactly.

--------------------------------------------------------------------------------
SECTION 2: Software Environment
--------------------------------------------------------------------------------

Operating System:    Windows 11 Professional
Python Version:      3.11.4
CUDA Version:        12.1
PyTorch Version:     2.0.1

================================================================================

    NOTICE: This document is formatted to standard 80-column terminal width.
    Each line should be exactly 80 characters or less for proper display.
    Lines that exceed this limit will wrap incorrectly on standard terminals.

================================================================================
"""

CODE_SAMPLES = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n

    # Initialize array to store Fibonacci numbers
    fib = [0] * (n + 1)
    fib[1] = 1

    # Build up the solution iteratively
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n]


class DataProcessor:
    """A class for processing and transforming data."""

    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.processed = False

    def load_data(self, filepath: str) -> None:
        """Load data from the specified file path."""
        with open(filepath, 'r') as f:
            self.data = f.read()
        print(f"Loaded {len(self.data)} bytes from {filepath}")

    def process(self) -> str:
        """Process the loaded data according to configuration."""
        if self.data is None:
            raise ValueError("No data loaded")

        result = self.data.upper()
        self.processed = True
        return result


# Main execution
if __name__ == "__main__":
    processor = DataProcessor({"mode": "standard"})
    processor.load_data("input.txt")
    output = processor.process()
    print(f"Processed {len(output)} characters")
'''

RFC_STYLE_TEXT = """
                            PROTOCOL SPECIFICATION

1. Introduction

   This document describes the format and procedures for data exchange
   between networked computer systems.  The protocol is designed to be
   simple, efficient, and reliable.

   The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
   "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
   document are to be interpreted as described in RFC 2119.

2. Protocol Overview

   2.1. Message Format

      All messages consist of a header followed by an optional body.
      The header contains routing information and metadata.
      The body contains the actual payload data.

      Header fields:
         - Version (1 byte): Protocol version number
         - Type (1 byte): Message type identifier
         - Length (2 bytes): Total message length
         - Sequence (4 bytes): Message sequence number

   2.2. Connection Establishment

      A connection is established through a three-way handshake:

         1. Client sends SYN
         2. Server responds with SYN-ACK
         3. Client sends ACK

      After this exchange, the connection is considered established.

3. Error Handling

   All errors MUST be reported using the standard error response format.
   The error code indicates the type of failure that occurred.
   Additional details MAY be included in the error message field.
"""


def get_real_text_lines(min_length: int = 10, max_length: int = 100) -> List[Tuple[str, int]]:
    """
    Get real text lines with their lengths.
    Returns list of (line, length) tuples.
    """
    all_text = FIXED_WIDTH_SAMPLES + CODE_SAMPLES + RFC_STYLE_TEXT

    lines = []
    for line in all_text.split('\n'):
        # Keep the line as-is (preserve original formatting)
        length = len(line)
        if min_length <= length <= max_length:
            lines.append((line, length))

    return lines


def fetch_gutenberg_text(url: str) -> str:
    """Fetch text from Project Gutenberg."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


# ============================================================
# Embedding Extraction
# ============================================================

def extract_embeddings(model, tokenizer, lines, layer=-1, device="cpu"):
    model = model.to(device)
    model.eval()
    embeddings = []
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512)
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
# HTML Visualization
# ============================================================

def create_html_visualization(
    lines_with_predictions: List[Dict],
    threshold: int,
    accuracy: float,
    model_name: str,
    output_file: str
):
    """Create HTML visualization of probe predictions."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Line Length Probe - {model_name}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #1e1e1e;
            color: #d4d4d4;
        }}
        h1 {{ color: #569cd6; text-align: center; }}
        h2 {{ color: #9cdcfe; margin-top: 30px; }}
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
        .model-info {{
            color: #ce9178;
            font-weight: bold;
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
            white-space: pre;
            overflow-x: auto;
            font-size: 0.9em;
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
            font-size: 0.8em;
        }}
        .analysis {{
            margin: 20px 0;
            padding: 15px;
            background: #252526;
            border-radius: 8px;
            border-left: 4px solid #569cd6;
        }}
    </style>
</head>
<body>
    <h1>Line Length Probe Validation</h1>

    <div class="stats">
        <div class="model-info">Model: {model_name}</div>
        <div class="threshold-info">Threshold: {threshold} characters</div>
        <div class="threshold-info">Task: Predict if line length &le; {threshold} (short) or &gt; {threshold} (long)</div>
        <div style="margin-top: 15px;">
            Validation Accuracy: <span class="accuracy">{accuracy:.1%}</span>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #4ec9b0;"></div>
            <span>Short (&le; {threshold} chars)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ce9178;"></div>
            <span>Long (&gt; {threshold} chars)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f14c4c;"></div>
            <span>Wrong prediction</span>
        </div>
    </div>

    <div class="ruler">
        <div>{"".join([str(i % 10) for i in range(1, 101)])}</div>
        <div>{"".join(["│" if i % 10 == 0 else " " for i in range(1, 101)])}</div>
        <div>{"".join([" " for _ in range(threshold-1)])}<span style="color: #f14c4c;">↑ threshold</span></div>
    </div>
"""

    # Analyze errors
    errors_by_length = {}
    for item in lines_with_predictions:
        if item['predicted'] != item['actual']:
            bucket = (item['length'] // 10) * 10
            errors_by_length[bucket] = errors_by_length.get(bucket, 0) + 1

    if errors_by_length:
        html += """
    <div class="analysis">
        <h3>Error Analysis</h3>
        <p>Errors by line length bucket:</p>
        <ul>
"""
        for bucket in sorted(errors_by_length.keys()):
            html += f"            <li>{bucket}-{bucket+9} chars: {errors_by_length[bucket]} errors</li>\n"
        html += """        </ul>
    </div>
"""

    html += "\n    <h2>Individual Predictions</h2>\n"

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

        # Escape HTML characters
        line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        html += f"""
    <div class="line-container {status_class}">
        <div class="line-text {text_class}">{line_escaped}</div>
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

def run_experiment(model_name: str, threshold: int = 50, device: str = "cpu"):
    """Run line length probe experiment."""

    print(f"\n{'='*70}")
    print(f"LINE LENGTH PROBE EXPERIMENT")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold} chars")
    print(f"{'='*70}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get lines from fixed-width formatted text
    print("\nPreparing data from fixed-width formatted text...")
    all_lines = get_real_text_lines(min_length=5, max_length=100)

    # Create labels
    lines_with_labels = [
        (line, length, 0 if length <= threshold else 1)
        for line, length in all_lines
    ]

    # Shuffle and split
    random.shuffle(lines_with_labels)

    # Split 70/30 for train/validation
    split_idx = int(len(lines_with_labels) * 0.7)
    train_data = lines_with_labels[:split_idx]
    val_data = lines_with_labels[split_idx:]

    train_lines = [x[0] for x in train_data]
    train_labels = torch.tensor([x[2] for x in train_data])
    train_lengths = [x[1] for x in train_data]

    val_lines = [x[0] for x in val_data]
    val_labels = [x[2] for x in val_data]
    val_lengths = [x[1] for x in val_data]

    print(f"  Training samples: {len(train_lines)}")
    print(f"    Short (<= {threshold}): {(train_labels == 0).sum().item()}")
    print(f"    Long (> {threshold}): {(train_labels == 1).sum().item()}")
    print(f"  Validation samples: {len(val_lines)}")
    print(f"    Short: {sum(1 for l in val_labels if l == 0)}")
    print(f"    Long: {sum(1 for l in val_labels if l == 1)}")

    # Extract training embeddings
    print("\nExtracting training embeddings...")
    train_embeddings = extract_embeddings(model, tokenizer, train_lines, layer=-1, device=device)

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

    # Training accuracy
    train_preds = classifier.predict(train_proj_scaled)
    train_acc = (train_preds == train_labels.numpy()).mean()
    print(f"\nTraining Accuracy: {train_acc:.1%}")

    # Extract validation embeddings
    print("\nExtracting validation embeddings...")
    val_embeddings = extract_embeddings(model, tokenizer, val_lines, layer=-1, device=device)

    # Project and predict
    with torch.no_grad():
        val_projected = probe(val_embeddings)

    val_proj_scaled = scaler.transform(val_projected.numpy())
    predictions = classifier.predict(val_proj_scaled)
    probabilities = classifier.predict_proba(val_proj_scaled)

    # Calculate accuracy
    correct = sum(p == a for p, a in zip(predictions, val_labels))
    accuracy = correct / len(val_labels)
    print(f"Validation Accuracy: {accuracy:.1%} ({correct}/{len(val_labels)})")

    # Prepare data for visualization
    lines_with_predictions = []
    for i, (line, length, actual) in enumerate(val_data):
        lines_with_predictions.append({
            'line': line,
            'length': length,
            'predicted': predictions[i],
            'actual': actual,
            'probability': probabilities[i][predictions[i]]
        })

    # Create HTML visualization
    safe_model_name = model_name.replace('/', '_')
    output_file = f"line_length_real_{safe_model_name}_{threshold}.html"
    create_html_visualization(
        lines_with_predictions,
        threshold,
        accuracy,
        model_name,
        output_file
    )

    return accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--threshold", type=int, default=50, help="Line length threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare", action="store_true", help="Compare GPT-2 and Qwen")
    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        models_to_test = [
            "gpt2",
            # "Qwen/Qwen2.5-1.5B",  # Smaller Qwen for faster testing
        ]

        results = {}
        for model_name in models_to_test:
            print(f"\n{'='*70}")
            print(f"TESTING: {model_name}")
            print(f"{'='*70}")
            try:
                acc = run_experiment(model_name, threshold=args.threshold, device=args.device)
                results[model_name] = acc
            except Exception as e:
                print(f"Failed: {e}")
                results[model_name] = None

        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        for model, acc in results.items():
            if acc is not None:
                print(f"{model}: {acc:.1%}")
            else:
                print(f"{model}: FAILED")
    else:
        # Single model test
        run_experiment(args.model, threshold=args.threshold, device=args.device)


if __name__ == "__main__":
    main()

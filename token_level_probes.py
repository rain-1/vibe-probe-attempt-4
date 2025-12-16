"""
Token-level line length probe visualization with proper buckets and tooltips.

For every token, show predicted line length bucket vs ground truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import argparse
from pathlib import Path
import random
import html as html_module

# ============================================================
# Fixed-Width Text
# ============================================================

SAMPLE_TEXT = """================================================================================
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

# Line length buckets
BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BUCKET_COLORS = ["#1f6feb", "#238636", "#d1240a", "#a371f7", "#fb8500"]
BUCKET_LABELS = ["0-20ch", "20-40ch", "40-60ch", "60-80ch", "80-100ch"]


def get_bucket(length: int) -> int:
    """Get bucket index for a line length."""
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return len(BUCKETS) - 1


class LengthRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)


def train_regression_probe(model, tokenizer, text_sources: List[str], device: str = "cuda"):
    """Train a regression probe to predict position in current line from mixed sources."""
    
    print(f"Extracting training data (token positions in lines) from {len(text_sources)} sources...")
    
    # Extract token positions from all text sources
    embeddings_list = []
    positions_list = []
    
    model = model.to(device)
    model.eval()
    
    for src_idx, text in enumerate(text_sources):
        print(f"  Source {src_idx + 1}/{len(text_sources)}: {len(text)} chars")
        
        # Process text to get token-level positions
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
        input_ids = inputs["input_ids"][0]
        offset_mapping = inputs["offset_mapping"][0]
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.unsqueeze(0).to(device),
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
        
        for i, (token_id, (start, end)) in enumerate(zip(input_ids, offset_mapping)):
            start_pos = start.item()
            
            # Find which line this token is on
            line_num, pos_in_line, _ = get_line_at_position(text, start_pos)
            
            emb = hidden_states[i].cpu()
            embeddings_list.append(emb)
            positions_list.append(float(pos_in_line))
    
    embeddings = torch.stack(embeddings_list)
    positions = np.array(positions_list, dtype=np.float32)
    
    print(f"Found {len(embeddings)} tokens with positions")
    print(f"  Position range: {positions.min():.0f} to {positions.max():.0f} chars")
    
    input_dim = embeddings.shape[1]
    
    print(f"\nTraining position regression probe (mixed data)...")
    
    # Train regression probe
    probe = LengthRegressor(input_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    probe = probe.to(device)
    for epoch in range(100):
        perm = torch.randperm(len(embeddings))
        for i in range(0, len(embeddings), 32):
            batch_idx = perm[i:i+32]
            batch_emb = embeddings[batch_idx].to(device)
            batch_pos = torch.tensor(positions[batch_idx], dtype=torch.float32).to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            pred = probe(batch_emb)
            loss = criterion(pred, batch_pos)
            loss.backward()
            optimizer.step()
    
    # Evaluate on training data
    probe.eval()
    with torch.no_grad():
        train_preds = probe(embeddings.to(device)).cpu().squeeze().numpy()
    
    train_mse = np.mean((train_preds - positions) ** 2)
    train_mae = np.mean(np.abs(train_preds - positions))
    print(f"  Train MSE: {train_mse:.2f}")
    print(f"  Train MAE: {train_mae:.2f}")
    
    return probe


def get_line_at_position(text: str, char_pos: int) -> Tuple[int, int, str]:
    """Get line number, position in line, and line content at character position."""
    lines = text.split('\n')
    current_pos = 0
    for line_num, line in enumerate(lines):
        line_len = len(line) + 1  # +1 for newline
        if current_pos + line_len > char_pos:
            pos_in_line = char_pos - current_pos
            return line_num, pos_in_line, line
        current_pos += line_len
    return len(lines) - 1, 0, lines[-1]


def extract_token_embeddings(model, tokenizer, text: str, device: str = "cuda", max_chars: int = 25000):
    """Extract embeddings for each token in the text (process in chunks if needed)."""
    
    # For large texts, only process first portion
    if len(text) > max_chars:
        print(f"Text too large ({len(text)} chars), using first {max_chars} chars")
        text = text[:max_chars]
    
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    input_ids = inputs["input_ids"][0]
    offset_mapping = inputs["offset_mapping"][0]
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0).to(device),
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
    
    results = []
    for i, (token_id, (start, end)) in enumerate(zip(input_ids, offset_mapping)):
        token = tokenizer.decode([token_id])
        emb = hidden_states[i].cpu()
        
        # Get line info
        line_num, pos_in_line, line_content = get_line_at_position(text, start.item())
        actual_line_len = len(line_content)
        
        results.append({
            'token': token,
            'embedding': emb,
            'char_start': start.item(),
            'char_end': end.item(),
            'line_num': line_num,
            'pos_in_line': pos_in_line,
            'line_content': line_content,
            'actual_line_len': actual_line_len,
            'actual_bucket': get_bucket(actual_line_len)
        })
    
    return results


def apply_probe_to_tokens(token_data: List[Dict], probe, device: str = "cuda"):
    """Apply probe to predict position in line for each token."""
    probe = probe.to(device)
    probe.eval()
    
    for item in token_data:
        emb = item['embedding'].to(device)
        
        with torch.no_grad():
            pred_position = probe(emb.unsqueeze(0)).item()
        
        pred_position = max(0, min(100, pred_position))  # Clamp to [0, 100]
        
        item['predicted_pos'] = pred_position
        item['error'] = abs(pred_position - item['pos_in_line'])
    
    return token_data


def create_visualization(token_data: List[Dict], model_name: str, output_file: str):
    """Create HTML visualization of token-level position prediction."""
    
    # Calculate stats
    all_errors = [item['error'] for item in token_data]
    avg_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    
    # Find max position for scaling
    max_pos = max(item['pos_in_line'] for item in token_data)
    
    def pos_to_color(pos: float, max_pos: float) -> str:
        """Map position to color gradient (blue -> red)."""
        norm = pos / max(1, max_pos)
        r = int(255 * norm)
        b = int(255 * (1 - norm))
        return f"rgb({r}, 100, {b})"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Token-Level Position in Line Prediction - {model_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #161b22;
            border-radius: 12px;
            padding: 30px;
        }}
        h1 {{
            text-align: center;
            color: #79c0ff;
            margin-bottom: 5px;
            font-size: 2.2em;
        }}
        .subtitle {{
            text-align: center;
            color: #8b949e;
            margin-bottom: 30px;
            font-size: 0.95em;
        }}
        .gradient-bar {{
            display: flex;
            height: 30px;
            border-radius: 6px;
            margin: 20px 0;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        .gradient-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: #8b949e;
            margin-bottom: 5px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        .stat {{
            background: #0d1117;
            padding: 18px;
            border-radius: 8px;
            border-left: 3px solid #58a6ff;
            text-align: center;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 0.85em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .stat-value {{
            color: #79c0ff;
            font-size: 1.8em;
            font-weight: 600;
        }}
        .text-container {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 25px;
            margin: 25px 0;
            word-wrap: break-word;
            line-height: 2.8;
            font-family: 'Courier New', monospace;
        }}
        .line-header {{
            margin-top: 20px;
            margin-bottom: 12px;
            padding: 8px 12px;
            background: #1c2128;
            border-radius: 4px;
            font-size: 0.75em;
            color: #6e7681;
            border-left: 3px solid #58a6ff;
        }}
        .token-wrapper {{
            position: relative;
            display: inline-block;
        }}
        .token {{
            display: inline-block;
            padding: 6px 8px;
            margin: 2px 1px;
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.15s ease;
            border: 1px solid rgba(88, 166, 255, 0.3);
            cursor: help;
        }}
        .token:hover {{
            transform: scale(1.1);
            border: 1px solid #58a6ff;
            box-shadow: 0 0 12px rgba(88, 166, 255, 0.4);
        }}
        .tooltip {{
            visibility: hidden;
            position: absolute;
            bottom: 130%;
            left: 50%;
            transform: translateX(-50%);
            background: #1c2128;
            color: #c9d1d9;
            text-align: left;
            padding: 12px 14px;
            border-radius: 6px;
            z-index: 1000;
            border: 1px solid #58a6ff;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            font-size: 0.8em;
            line-height: 1.5;
            white-space: pre-line;
            pointer-events: none;
        }}
        .token:hover .tooltip {{
            visibility: visible;
        }}
        .tooltip-line {{
            margin: 4px 0;
        }}
        .tooltip-label {{
            font-weight: 700;
            color: #79c0ff;
            display: inline-block;
            min-width: 110px;
        }}
        .error-good {{ color: #3fb950; font-weight: bold; }}
        .error-ok {{ color: #d29922; font-weight: bold; }}
        .error-bad {{ color: #f85149; font-weight: bold; }}
        .info-box {{
            background: #0d1117;
            border-left: 3px solid #79c0ff;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 0.9em;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Token Position in Line Prediction</h1>
        <p class="subtitle">Model: {model_name} | How many characters into the line is each token?</p>
        
        <div class="info-box">
            <strong>What this shows:</strong> Each token is colored by its <strong>predicted position in the current line</strong> 
            (characters since the last newline). 
            <strong>Hover over any token</strong> to see the actual position, prediction, and error. 
            The color gradient goes from <strong style="color: #79c0ff;">blue (start of line)</strong> to <strong style="color: #f85149;">red (end of line)</strong>.
        </div>
        
        <div class="gradient-label">
            <span>Start of line (0ch)</span>
            <span>End of line ({max_pos:.0f}ch)</span>
        </div>
        <div class="gradient-bar" style="background: linear-gradient(90deg, rgb(0, 100, 255) 0%, rgb(255, 100, 0) 100%);"></div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-value">{len(token_data)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Lines</div>
                <div class="stat-value">{max(item['line_num'] for item in token_data) + 1}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Avg Error</div>
                <div class="stat-value">{avg_error:.1f} ch</div>
            </div>
            <div class="stat">
                <div class="stat-label">Median Error</div>
                <div class="stat-value">{median_error:.1f} ch</div>
            </div>
        </div>
        
        <div class="text-container">
"""
    
    current_line = -1
    for item in token_data:
        if item['line_num'] != current_line:
            current_line = item['line_num']
            line_len = item['actual_line_len']
            html_content += f"""            <div class="line-header">
                Line {current_line + 1} | Length: <strong>{line_len} characters</strong>
            </div>
"""
        
        token_text = html_module.escape(item['token'])
        actual_pos = item['pos_in_line']
        pred_pos = item['predicted_pos']
        error = item['error']
        
        # Error class for coloring
        if error <= 2:
            error_class = "error-good"
        elif error <= 5:
            error_class = "error-ok"
        else:
            error_class = "error-bad"
        
        color = pos_to_color(pred_pos, max_pos)
        
        html_content += f"""            <div class="token-wrapper">
                <span class="token" style="background: {color}; opacity: 0.8;">
                    {token_text}
                    <div class="tooltip">
<span class="tooltip-line"><span class="tooltip-label">Position:</span> {actual_pos:.0f} chars</span>
<span class="tooltip-line"><span class="tooltip-label">Predicted:</span> {pred_pos:.1f} chars</span>
<span class="tooltip-line"><span class="tooltip-label">Error:</span> <span class="{error_class}">{error:.1f} chars</span></span>
                    </div>
                </span>
            </div>"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n✅ Saved visualization to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get text sources for training (mix of both)
    training_texts = [SAMPLE_TEXT]  # Always include formatted text
    if args.text_file:
        # Also include the provided file (first chunk)
        file_text = Path(args.text_file).read_text(encoding='utf-8')[:25000]
        training_texts.append(file_text)
        test_text = file_text
    else:
        test_text = SAMPLE_TEXT
    
    print(f"\nTraining on {len(training_texts)} text source(s)...")
    probe = train_regression_probe(model, tokenizer, training_texts, device=args.device)
    
    print(f"\nExtracting token embeddings from test text ({len(test_text)} chars)...")
    token_data = extract_token_embeddings(model, tokenizer, test_text, device=args.device)
    print(f"✅ Extracted {len(token_data)} tokens")
    
    print(f"\nApplying probe to tokens...")
    token_data = apply_probe_to_tokens(token_data, probe, device=args.device)
    
    if args.text_file:
        safe_model = args.model.replace("/", "_")
        output_file = args.output or f"visualizations/token_mixed_{safe_model}.html"
    else:
        safe_model = args.model.replace("/", "_")
        output_file = args.output or f"visualizations/token_line_length_{safe_model}.html"
    
    print(f"\nGenerating visualization...")
    create_visualization(token_data, args.model, output_file)


if __name__ == "__main__":
    main()

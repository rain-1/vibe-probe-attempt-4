"""
Visualize trained code probes on code samples.

Creates an interactive HTML visualization showing probe activations
across different semantic categories (keywords, strings, comments, etc.)
"""

import argparse
import html
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LinearProbe(nn.Module):
    """Simple linear probe for binary or multi-class classification."""

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes if num_classes > 2 else 1)
        self.num_classes = num_classes

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize code probe activations")
    parser.add_argument(
        "--probes",
        type=str,
        nargs="+",
        required=True,
        help="Probe checkpoint files (can specify multiple)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-pt",
        help="Model to use",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="code-data-work/tokenized_data/combined.jsonl",
        help="Data file to sample from",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Specific sample index to visualize (random if not specified)",
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Custom code to visualize (overrides --data)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a source file to visualize (overrides --data).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/code_probe_viz.html",
        help="Output HTML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to include in visualization",
    )
    return parser.parse_args()


def load_probes(probe_paths: list[str], device: str):
    """Load probe checkpoints."""
    probes = {}
    for path in probe_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        target = checkpoint["target"]
        layer = checkpoint["layer"]
        input_dim = checkpoint["input_dim"]
        num_classes = checkpoint.get("num_classes", 2)

        probe = LinearProbe(input_dim, num_classes)
        probe.load_state_dict(checkpoint["model_state_dict"])
        probe = probe.to(device)
        probe.eval()

        key = f"{target}_L{layer}"
        probes[key] = {
            "probe": probe,
            "target": target,
            "layer": layer,
            "num_classes": num_classes,
        }
        print(f"Loaded probe: {key} (classes={num_classes})")

    return probes


def load_sample(data_path: str, idx: int = None):
    """Load a sample from the data file."""
    with open(data_path, "r", encoding="utf-8") as f:
        if idx is not None:
            for i, line in enumerate(f):
                if i == idx:
                    return json.loads(line)
            raise ValueError(f"Index {idx} out of range")
        else:
            lines = f.readlines()
            line = random.choice(lines)
            return json.loads(line)


def load_multiple_samples(data_path: str, num_samples: int = 5):
    """Load multiple random samples from the data file."""
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    indices = random.sample(range(len(lines)), min(num_samples, len(lines)))
    samples = []
    for idx in indices:
        sample = json.loads(lines[idx])
        sample["_idx"] = idx
        samples.append(sample)
    return samples


def get_hidden_states(model, input_ids, layers: list, device: str):
    """Get hidden states for specified layers."""
    inputs = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True, return_dict=True)

    hidden_states = {}
    for layer in layers:
        if layer == "final":
            # Post-layer-norm
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                last_hidden = outputs.hidden_states[-1][0]
                normed = model.model.norm(last_hidden)
                hidden_states[layer] = normed.cpu().float()
            else:
                hidden_states[layer] = outputs.hidden_states[-1][0].cpu().float()
        else:
            hidden_states[layer] = outputs.hidden_states[layer][0].cpu().float()

    return hidden_states


def compute_activations(probes: dict, hidden_states: dict, device: str):
    """Compute probe activations for each token."""
    activations = {}

    for key, probe_info in probes.items():
        probe = probe_info["probe"]
        layer = probe_info["layer"]
        num_classes = probe_info["num_classes"]

        if layer not in hidden_states:
            continue

        hidden = hidden_states[layer].to(device)

        with torch.no_grad():
            logits = probe(hidden)

            if num_classes <= 2:
                probs = torch.sigmoid(logits).cpu().tolist()
            else:
                # Multi-class: return class probabilities
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

        activations[key] = probs

    return activations


def generate_html(samples_data: list, output_path: str):
    """Generate interactive HTML visualization."""

    # Prepare data for JavaScript
    samples_json = json.dumps(samples_data)

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Probe Visualization</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: #1e1e2e;
            color: #cdd6f4;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #89b4fa;
            margin-bottom: 30px;
        }
        .controls {
            background: #313244;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-weight: bold;
            color: #f5c2e7;
        }
        select, button {
            padding: 8px 15px;
            border-radius: 5px;
            border: none;
            background: #45475a;
            color: #cdd6f4;
            font-size: 14px;
            cursor: pointer;
        }
        select:hover, button:hover {
            background: #585b70;
        }
        .code-container {
            background: #181825;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: 14px;
        }
        .token {
            display: inline;
            padding: 2px 0;
            border-radius: 3px;
            transition: all 0.2s ease;
        }
        .token:hover {
            outline: 2px solid #f5c2e7;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: #313244;
            border-radius: 10px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 40px;
            height: 20px;
            border-radius: 3px;
        }
        .info-panel {
            background: #313244;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-panel h3 {
            color: #a6e3a1;
            margin: 0 0 10px 0;
        }
        .info-panel p {
            margin: 5px 0;
            color: #bac2de;
        }
        .ground-truth-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ground-truth-toggle input {
            width: 18px;
            height: 18px;
        }
        .sample-nav {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .sample-nav button {
            padding: 8px 20px;
        }
        .sample-info {
            color: #f5c2e7;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Code Probe Visualization</h1>

    <div class="controls">
        <div class="sample-nav">
            <button id="prev-sample">Previous</button>
            <span class="sample-info" id="sample-info">Sample 1 / 1</span>
            <button id="next-sample">Next</button>
        </div>

        <div class="control-group">
            <label>Probe:</label>
            <select id="probe-select"></select>
        </div>

        <div class="ground-truth-toggle">
            <input type="checkbox" id="show-ground-truth">
            <label for="show-ground-truth">Show Ground Truth</label>
        </div>
    </div>

    <div class="info-panel" id="info-panel">
        <h3>Sample Info</h3>
        <p id="language-info">Language: -</p>
        <p id="tokens-info">Tokens: -</p>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(166, 227, 161, 0.1);"></div>
            <span>Low (0.0)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(249, 226, 175, 0.5);"></div>
            <span>Medium (0.5)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(243, 139, 168, 0.9);"></div>
            <span>High (1.0)</span>
        </div>
    </div>

    <div class="code-container" id="code-container"></div>

    <script>
        const samplesData = ''' + samples_json + ''';

        let currentSampleIdx = 0;
        let currentProbe = null;
        let showGroundTruth = false;

        // Populate probe selector
        const probeSelect = document.getElementById('probe-select');
        const probeKeys = Object.keys(samplesData[0].activations || {});
        probeKeys.forEach(key => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = key;
            probeSelect.appendChild(option);
        });
        currentProbe = probeKeys[0] || null;

        function getColor(value, isGroundTruth = false) {
            if (isGroundTruth) {
                // Ground truth: bright green for 1, no color for 0
                if (value > 0.5) {
                    return 'rgba(166, 227, 161, 0.8)';
                }
                return 'transparent';
            }

            // Probe activation: gradient from green to yellow to red
            const intensity = Math.min(1, Math.max(0, value));
            if (intensity < 0.5) {
                // Green to Yellow
                const r = Math.round(166 + (249 - 166) * (intensity * 2));
                const g = Math.round(227 + (226 - 227) * (intensity * 2));
                const b = Math.round(161 + (175 - 161) * (intensity * 2));
                return `rgba(${r}, ${g}, ${b}, ${0.1 + intensity * 0.5})`;
            } else {
                // Yellow to Red
                const t = (intensity - 0.5) * 2;
                const r = Math.round(249 + (243 - 249) * t);
                const g = Math.round(226 + (139 - 226) * t);
                const b = Math.round(175 + (168 - 175) * t);
                return `rgba(${r}, ${g}, ${b}, ${0.5 + t * 0.4})`;
            }
        }

        function updateVisualization() {
            const sample = samplesData[currentSampleIdx];
            const container = document.getElementById('code-container');

            // Update info
            document.getElementById('sample-info').textContent =
                `Sample ${currentSampleIdx + 1} / ${samplesData.length} (idx: ${sample.idx})`;
            document.getElementById('language-info').textContent =
                `Language: ${sample.language}`;
            document.getElementById('tokens-info').textContent =
                `Tokens: ${sample.tokens.length}`;

            // Get activations and ground truth
            const activations = currentProbe ? (sample.activations[currentProbe] || []) : [];

            // Extract ground truth target name from probe key (e.g., "is_keyword_L12" -> "is_keyword")
            let groundTruth = [];
            if (showGroundTruth && currentProbe) {
                const targetName = currentProbe.split('_L')[0];
                groundTruth = sample.ground_truth[targetName] || [];
            }

            // Render tokens
            let html = '';
            sample.tokens.forEach((token, i) => {
                let value = activations[i];
                // Handle multi-class (array of probs)
                if (Array.isArray(value)) {
                    // Use max non-zero class prob
                    value = Math.max(...value.slice(1));
                }
                value = value || 0;

                let gtValue = groundTruth[i] || 0;
                // Handle categorical ground truth
                if (typeof gtValue === 'number' && gtValue > 0) {
                    gtValue = 1;
                }

                const color = showGroundTruth ? getColor(gtValue, true) : getColor(value, false);
                const tooltip = `Token: ${token}\\nProbe: ${typeof activations[i] === 'number' ? activations[i].toFixed(3) : JSON.stringify(activations[i])}\\nGT: ${groundTruth[i] || 0}`;

                // Escape HTML in token
                const escapedToken = token
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');

                html += `<span class="token" style="background-color: ${color};" title="${tooltip}">${escapedToken}</span>`;
            });

            container.innerHTML = html;
        }

        // Event listeners
        probeSelect.addEventListener('change', (e) => {
            currentProbe = e.target.value;
            updateVisualization();
        });

        document.getElementById('show-ground-truth').addEventListener('change', (e) => {
            showGroundTruth = e.target.checked;
            updateVisualization();
        });

        document.getElementById('prev-sample').addEventListener('click', () => {
            currentSampleIdx = Math.max(0, currentSampleIdx - 1);
            updateVisualization();
        });

        document.getElementById('next-sample').addEventListener('click', () => {
            currentSampleIdx = Math.min(samplesData.length - 1, currentSampleIdx + 1);
            updateVisualization();
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                currentSampleIdx = Math.max(0, currentSampleIdx - 1);
                updateVisualization();
            } else if (e.key === 'ArrowRight') {
                currentSampleIdx = Math.min(samplesData.length - 1, currentSampleIdx + 1);
                updateVisualization();
            }
        });

        // Initial render
        updateVisualization();
    </script>
</body>
</html>'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\nVisualization saved to: {output_path}")


def main():
    args = parse_args()

    # Load probes
    probes = load_probes(args.probes, args.device)
    if not probes:
        print("ERROR: No probes loaded")
        return

    # Get unique layers needed
    layers = list(set(p["layer"] for p in probes.values()))

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    samples_data = []

    if args.code:
        # Custom code provided
        input_ids = tokenizer.encode(args.code)
        tokens = [tokenizer.decode([tid]) for tid in input_ids]

        hidden_states = get_hidden_states(model, input_ids, layers, args.device)
        activations = compute_activations(probes, hidden_states, args.device)

        samples_data.append({
            "idx": -1,
            "language": "custom",
            "tokens": tokens,
            "activations": activations,
            "ground_truth": {},
        })
    elif args.file:
        # Load code from a provided file path
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: file not found: {file_path}")
            return
        code_text = file_path.read_text(encoding="utf-8")
        input_ids = tokenizer.encode(code_text)
        tokens = [tokenizer.decode([tid]) for tid in input_ids]

        hidden_states = get_hidden_states(model, input_ids, layers, args.device)
        activations = compute_activations(probes, hidden_states, args.device)

        samples_data.append({
            "idx": -1,
            "language": file_path.suffix.lstrip('.') or "file",
            "tokens": tokens,
            "activations": activations,
            "ground_truth": {},
        })

    else:
        # Load samples from data file
        print(f"Loading samples from: {args.data}")
        samples = load_multiple_samples(args.data, args.num_samples)

        for sample in samples:
            input_ids = sample["input_ids"][:256]  # Limit length
            tokens = [tokenizer.decode([tid]) for tid in input_ids]

            hidden_states = get_hidden_states(model, input_ids, layers, args.device)
            activations = compute_activations(probes, hidden_states, args.device)

            # Extract ground truth
            ground_truth = {}
            for key in ["is_number", "is_comment", "is_string", "is_keyword",
                        "is_operator", "is_punctuation", "is_identifier",
                        "var_scope", "var_type"]:
                if key in sample:
                    ground_truth[key] = sample[key][:len(input_ids)]

            samples_data.append({
                "idx": sample.get("_idx", -1),
                "language": sample.get("language", "unknown"),
                "tokens": tokens,
                "activations": activations,
                "ground_truth": ground_truth,
            })

    # Generate HTML
    generate_html(samples_data, args.output)
    print("Done!")


if __name__ == "__main__":
    main()

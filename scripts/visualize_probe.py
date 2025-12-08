"""
Probe visualizer - generates HTML visualization of probe activations across tokens.

Generates text using the same warm-up + low-temp strategy, then attaches trained
probes to visualize activation patterns across layers.
"""

import argparse
import html
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize probe activations across tokens"
    )
    parser.add_argument(
        "--probes",
        type=str,
        nargs="+",
        required=True,
        help="Probe directory or list of checkpoint files (e.g., --probes checkpoints/ or --probes probe1.pt probe2.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-270m",
        help="Model to use (default: unsloth/gemma-3-270m)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to visualize (if not provided, generates new text)",
    )
    parser.add_argument(
        "--warm-tokens",
        type=int,
        default=20,
        help="Warm-up tokens at temp=1.0 for generation (default: 20)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=100,
        help="Tokens to generate at temp=0 (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/probe_viz.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, device: str, gen_tokens: int) -> tuple[str, list[int]]:
    """
    Generate text at temp=1.0.
    Returns (text, token_ids).
    """
    input_ids = tokenizer.encode("The", return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=gen_tokens,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get generated tokens (excluding prompt)
    generated_tokens = output[0, 1:]  # Skip the "The" prompt token
    token_ids = generated_tokens.tolist()
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return text, token_ids


def load_probes(probe_paths: list[str], device: str) -> dict[int | str, nn.Module]:
    """Load probe checkpoints and return dict mapping layer -> probe."""
    probes = {}
    for path in probe_paths:
        checkpoint = torch.load(path, map_location=device)
        layer = checkpoint["layer"]  # Can be int or 'final'
        input_dim = checkpoint["input_dim"]
        use_gelu = checkpoint.get("use_gelu", False)
        
        probe = LinearProbe(input_dim, use_gelu=use_gelu)
        probe.load_state_dict(checkpoint["model_state_dict"])
        probe = probe.to(device)
        probe.eval()
        
        probes[layer] = probe
        print(f"Loaded probe for layer {layer} from {path}")
    
    return probes


def get_all_hidden_states(model, tokenizer, text: str, device: str) -> dict:
    """
    Run model on text and extract hidden states from all layers for all tokens.
    Returns dict with layer -> (seq_len, hidden_dim) tensor.
    Also includes 'final' key for post-layer-norm hidden states.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden_dim)
    
    result = {}
    for layer_idx, layer_hidden in enumerate(hidden_states):
        result[layer_idx] = layer_hidden[0]  # Remove batch dim -> (seq, hidden_dim)
    
    # Add post-final-layer-norm hidden states
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm  # Gemma, Llama style
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        final_norm = model.transformer.ln_f  # GPT-2 style
    
    if final_norm is not None:
        last_hidden = hidden_states[-1][0]  # (seq, hidden_dim)
        with torch.no_grad():
            normed_hidden = final_norm(last_hidden)
        result["final"] = normed_hidden
    
    return result, inputs.input_ids[0].tolist()


def compute_probe_activations(probes: dict[int | str, nn.Module], hidden_states: dict) -> dict[int | str, list[float]]:
    """
    Apply each probe to its corresponding layer's hidden states.
    Returns dict mapping layer -> list of activation values per token.
    """
    activations = {}
    
    for layer, probe in probes.items():
        if layer not in hidden_states:
            print(f"Warning: Layer {layer} not in hidden states, skipping")
            continue
        
        layer_hidden = hidden_states[layer]  # (seq, hidden_dim)
        
        with torch.no_grad():
            # Apply probe to each token position
            logits = probe(layer_hidden.float())  # (seq,)
            probs = torch.sigmoid(logits)  # Convert to probability
            activations[layer] = probs.cpu().tolist()
    
    return activations


def generate_html(tokens: list[str], activations: dict[int | str, list[float]], output_path: str):
    """
    Generate HTML visualization with layer slider.
    """
    # Sort layers: integers first (numerically), then strings
    int_layers = sorted([l for l in activations.keys() if isinstance(l, int)])
    str_layers = sorted([l for l in activations.keys() if isinstance(l, str)])
    layers = int_layers + str_layers
    
    # Convert activations to JSON for JavaScript (ensure keys are strings)
    activations_str_keys = {str(k): v for k, v in activations.items()}
    activations_json = json.dumps(activations_str_keys)
    tokens_json = json.dumps([html.escape(t) for t in tokens])
    layers_json = json.dumps([str(l) for l in layers])
    
    # Default to middle layer for initial view
    default_idx = len(layers) // 2
    default_layer = str(layers[default_idx])
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Probe Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
        }}
        .controls {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .slider-container label {{
            font-weight: bold;
            min-width: 100px;
        }}
        #layer-slider {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
            outline: none;
        }}
        #layer-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #layer-value {{
            font-size: 1.2em;
            color: #00d4ff;
            min-width: 80px;
            text-align: right;
        }}
        .text-container {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            line-height: 2;
            font-size: 16px;
        }}
        .token {{
            display: inline;
            padding: 2px 1px;
            border-radius: 3px;
            transition: background-color 0.3s ease;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 30px;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>🔬 Probe Activation Visualization</h1>
    
    <div class="controls">
        <div class="slider-container">
            <label for="layer-slider">Layer:</label>
            <input type="range" id="layer-slider" min="0" max="{len(layers) - 1}" value="{default_idx}" step="1">
            <span id="layer-value">Layer {default_layer}</span>
        </div>
    </div>
    
    <div class="text-container" id="text-container">
        <!-- Tokens will be inserted here -->
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(0, 100, 0);"></div>
            <span>Low (0.0)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(255, 255, 0);"></div>
            <span>Medium (0.5)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(255, 0, 0);"></div>
            <span>High (1.0)</span>
        </div>
    </div>
    
    <script>
        const tokens = {tokens_json};
        const activations = {activations_json};
        const layers = {layers_json};
        
        function getColor(value) {{
            // Green -> Yellow -> Red gradient
            if (value < 0.5) {{
                // Green to Yellow
                const r = Math.round(255 * (value * 2));
                const g = Math.round(100 + 155 * (value * 2));
                const b = 0;
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }} else {{
                // Yellow to Red
                const r = 255;
                const g = Math.round(255 * (1 - (value - 0.5) * 2));
                const b = 0;
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }}
        }}
        
        function updateVisualization(layer) {{
            const container = document.getElementById('text-container');
            const layerActivations = activations[layer];
            
            let html = '';
            for (let i = 0; i < tokens.length; i++) {{
                const value = layerActivations ? layerActivations[i] : 0;
                const color = getColor(value);
                const title = `Token: ${{tokens[i]}}\\nProbe: ${{value.toFixed(3)}}`;
                html += `<span class="token" style="background-color: ${{color}};" title="${{title}}">${{tokens[i]}}</span>`;
            }}
            container.innerHTML = html;
            document.getElementById('layer-value').textContent = `Layer ${{layer}}`;
        }}
        
        document.getElementById('layer-slider').addEventListener('input', function() {{
            const layerIdx = parseInt(this.value);
            const layer = layers[layerIdx];
            updateVisualization(layer);
        }});
        
        // Initial render
        updateVisualization(layers[{default_idx}]);
    </script>
</body>
</html>'''
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\nVisualization saved to: {output_path}")


def main():
    args = parse_args()
    
    # Expand probes argument - if single directory, find all .pt files
    probe_paths = []
    for p in args.probes:
        path = Path(p)
        if path.is_dir():
            # Find all .pt files in directory, sorted numerically by layer number
            import re
            found = list(path.glob("*.pt"))
            # Sort by layer number extracted from filename (e.g., probe_layer_12.pt -> 12)
            def extract_layer(f):
                match = re.search(r'(\d+)', f.stem)
                return int(match.group(1)) if match else 0
            found = sorted(found, key=extract_layer)
            probe_paths.extend([str(f) for f in found])
            print(f"Found {len(found)} probes in {path}")
        else:
            probe_paths.append(p)
    
    if not probe_paths:
        print("ERROR: No probe files found")
        return
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Load probes
    probes = load_probes(probe_paths, args.device)
    
    # Get or generate text
    if args.text:
        text = args.text
        print(f"Using provided text: {text[:100]}...")
    else:
        # Seed with current time for fresh output each run
        seed = int(time.time() * 1000) % (2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        print(f"Generating text (seed={seed})...")
        text, _ = generate_text(model, tokenizer, args.device, args.gen_tokens)
        print(f"Generated: {text[:100]}...")
    
    # Get hidden states for all layers
    print("Extracting hidden states...")
    hidden_states, token_ids = get_all_hidden_states(model, tokenizer, text, args.device)
    
    # Decode individual tokens for display
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"Tokens: {len(tokens)}")
    
    # Compute probe activations
    print("Computing probe activations...")
    activations = compute_probe_activations(probes, hidden_states)
    
    # Generate HTML
    generate_html(tokens, activations, args.output)
    print("Done!")


if __name__ == "__main__":
    main()

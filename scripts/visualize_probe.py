"""
Probe visualizer - generates HTML visualization of probe activations across tokens.

Generates text using the same warm-up + low-temp strategy, then attaches trained
probes to visualize activation patterns across layers.
"""

import argparse
import html
import json
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
        help="Paths to probe checkpoint files (one per layer)",
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


def generate_text(model, tokenizer, device: str, warm_tokens: int, gen_tokens: int) -> tuple[str, list[int]]:
    """
    Generate text using warm-up + low-temp strategy.
    Returns (text, token_ids).
    """
    input_ids = tokenizer.encode("The", return_tensors="pt").to(device)
    
    # Generate warm_tokens at temp=1.0
    with torch.no_grad():
        output_high_temp = model.generate(
            input_ids,
            max_new_tokens=warm_tokens,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Continue with gen_tokens at temp=0 (greedy)
    with torch.no_grad():
        output_full = model.generate(
            output_high_temp,
            max_new_tokens=gen_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Truncate to final gen_tokens
    final_tokens = output_full[0, -gen_tokens:]
    token_ids = final_tokens.tolist()
    text = tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    return text, token_ids


def load_probes(probe_paths: list[str], device: str) -> dict[int, nn.Module]:
    """Load probe checkpoints and return dict mapping layer -> probe."""
    probes = {}
    for path in probe_paths:
        checkpoint = torch.load(path, map_location=device)
        layer = checkpoint["layer"]
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
    
    return result, inputs.input_ids[0].tolist()


def compute_probe_activations(probes: dict[int, nn.Module], hidden_states: dict) -> dict[int, list[float]]:
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


def generate_html(tokens: list[str], activations: dict[int, list[float]], output_path: str):
    """
    Generate HTML visualization with layer slider.
    """
    layers = sorted(activations.keys())
    
    # Convert activations to JSON for JavaScript
    activations_json = json.dumps(activations)
    tokens_json = json.dumps([html.escape(t) for t in tokens])
    
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
            <input type="range" id="layer-slider" min="{min(layers)}" max="{max(layers)}" value="{layers[len(layers)//2]}" step="1">
            <span id="layer-value">Layer {layers[len(layers)//2]}</span>
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
        const layers = {json.dumps(layers)};
        
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
            updateVisualization(parseInt(this.value));
        }});
        
        // Initial render
        updateVisualization({layers[len(layers)//2]});
    </script>
</body>
</html>'''
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\nVisualization saved to: {output_path}")


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Load probes
    probes = load_probes(args.probes, args.device)
    
    # Get or generate text
    if args.text:
        text = args.text
        print(f"Using provided text: {text[:100]}...")
    else:
        print("Generating text...")
        text, _ = generate_text(model, tokenizer, args.device, args.warm_tokens, args.gen_tokens)
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

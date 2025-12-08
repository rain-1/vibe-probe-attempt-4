"""
Test a trained probe on arbitrary text.

This script loads a probe and runs it on text you provide, showing
for each token whether the probe predicts the target will come next.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Test a probe on text")
    parser.add_argument("--probe", type=str, required=True, help="Path to probe checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--target", type=str, default=" the", help="Target token (default: ' the')")
    parser.add_argument("--model", type=str, default="unsloth/gemma-3-270m", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(args.device)
    model.eval()
    
    # Load probe
    print(f"Loading probe: {args.probe}")
    ckpt = torch.load(args.probe, map_location=args.device)
    layer = ckpt["layer"]
    print(f"  Layer: {layer}")
    
    probe = LinearProbe(ckpt["input_dim"], use_gelu=ckpt.get("use_gelu", False))
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.to(args.device)
    probe.eval()
    
    # Get target token ID for comparison
    target_ids = tokenizer.encode(args.target, add_special_tokens=False)
    print(f"  Target: {repr(args.target)} -> token IDs {target_ids}")
    
    # Tokenize input
    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)
    token_ids = inputs.input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states for the probe's layer
        if layer == "final":
            # Post-layer-norm hidden states
            last_hidden = outputs.hidden_states[-1][0]
            hidden = model.model.norm(last_hidden)
        else:
            hidden = outputs.hidden_states[layer][0]
        
        # Run probe
        logits = probe(hidden.float())
        probs = torch.sigmoid(logits)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"{'Token':<15} {'Probe':>8} {'Next Token':<15} {'Result'}")
    print(f"{'='*70}")
    
    correct = 0
    total = 0
    
    for i, (tok, prob) in enumerate(zip(tokens, probs)):
        # Get actual next token
        if i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            next_id = token_ids[i + 1]
        else:
            next_tok = "[END]"
            next_id = None
        
        # Check if next token matches target
        is_target = next_id in target_ids if next_id is not None else False
        probe_predicts_target = prob.item() > 0.5
        
        # Determine correctness
        if is_target and probe_predicts_target:
            result = "✓ TRUE POSITIVE"
            correct += 1
        elif not is_target and not probe_predicts_target:
            result = "✓ true negative"
            correct += 1
        elif is_target and not probe_predicts_target:
            result = "✗ FALSE NEGATIVE"
        else:  # not is_target and probe_predicts_target
            result = "✗ FALSE POSITIVE"
        
        total += 1
        
        # Color the probability
        prob_str = f"{prob.item():.3f}"
        
        print(f"{repr(tok):<15} {prob_str:>8} {repr(next_tok):<15} {result}")
    
    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    main()

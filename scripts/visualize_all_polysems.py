"""
Apply all trained polysem probes/projections to a document and generate an HTML
where each occurrence of any polysemous word is annotated with predictions
from all available probes.

This script loads all probe files in `--probes-dir` matching `polysem_{word}_probe.pt`
and all projection files `polysem_contrastive_{word}_proj.pt` in the same dir.

Output: single HTML with occurrences colored and tooltips listing probe outputs.
"""
from pathlib import Path
import argparse
import re
import html
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_probe_files(dirpath: Path):
    probes = defaultdict(list)
    projs = {}
    for p in dirpath.glob("*.pt"):
        name = p.stem
        if name.startswith("polysem_") and name.endswith("_probe"):
            # polysem_<word>_probe
            word = name[len("polysem_"):-len("_probe")]
            probes[word].append(p)
        if name.startswith("polysem_contrastive_") and name.endswith("_proj"):
            word = name[len("polysem_contrastive_"):-len("_proj")]
            projs[word] = p
    return probes, projs


def load_linear_probe(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    # find linear.*weight & bias
    weight = None
    bias = None
    for k, v in state.items():
        if k.endswith("linear.weight") or k.endswith("weight"):
            weight = v
        if k.endswith("linear.bias") or k.endswith("bias"):
            bias = v
    if weight is None:
        raise RuntimeError(f"Cannot infer probe dims from {path}")
    out_dim, in_dim = weight.shape
    lin = torch.nn.Linear(in_dim, out_dim)
    lin.load_state_dict(state)
    lin = lin.to(device)
    lin.eval()
    return lin


def load_projection(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    W = ckpt.get("W")
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)
    return W


def extract_embedding_once(model, tokenizer, text_chunk: str, device, max_len=256):
    inputs = tokenizer(text_chunk, truncation=True, max_length=max_len, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1][0]
    vec = last_hidden.mean(dim=0)
    return vec.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--probes-dir", type=str, default="checkpoints/")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="visualizations/all_polysems.html")
    args = parser.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        print("No text-file provided; please pass --text-file")
        return

    probes_dir = Path(args.probes_dir)
    probes_map, projs_map = find_probe_files(probes_dir)
    words = set(list(probes_map.keys()) + list(projs_map.keys()))
    if not words:
        print("No probes or projections found in", probes_dir)
        return

    # Build regex for all words (word boundary, longest-first)
    sorted_words = sorted(words, key=lambda s: -len(s))
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted_words) + r")\b", flags=re.IGNORECASE)

    # Load model/tokenizer once
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # Load probes into memory
    loaded_probes = {}
    for w, plist in probes_map.items():
        loaded_probes[w] = [load_linear_probe(p, device) for p in plist]
    loaded_projs = {w: load_projection(p) for w, p in projs_map.items()}

    def score_to_color(v: float) -> str:
        # v expected in [0,1]
        v = max(0.0, min(1.0, v))
        if v < 0.5:
            r = int(255 * (v * 2))
            g = int(100 + 155 * (v * 2))
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (v - 0.5) * 2))
            b = 0
        return f"rgb({r},{g},{b})"

    # Iterate occurrences and apply probes
    out_parts = []
    last = 0
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        out_parts.append(html.escape(text[last:start]))
        word = m.group(0)
        lw = word.lower()

        # Use small context window
        left = max(0, start - 200)
        right = min(len(text), end + 200)
        context = text[left:right]

        emb = extract_embedding_once(model, tokenizer, context, device)
        emb_d = emb.to(device)

        tooltip_lines = []
        # Linear probes
        for p in loaded_probes.get(lw, []):
            with torch.no_grad():
                out = p(emb_d.unsqueeze(0))
                if out.dim() == 2 and out.size(1) > 1:
                    probs = F.softmax(out, dim=-1)[0].cpu().tolist()
                    pred = int(torch.argmax(out, dim=-1).item())
                    tooltip_lines.append(f"probe:{p.__class__.__name__} pred={pred} probs={probs}")
                else:
                    prob = torch.sigmoid(out).item()
                    tooltip_lines.append(f"probe:{p.__class__.__name__} score={prob:.3f}")

        # Projections
        if lw in loaded_projs:
            W = loaded_projs[lw]
            z = emb @ W.T
            # For evaluation, just show norm or first dims
            tooltip_lines.append(f"proj_sim_norm={z.norm().item():.3f}")

        tooltip = " | ".join(tooltip_lines) if tooltip_lines else "no-probe"

        # Decide color based on probes/projections
        color = None
        # Prefer linear probe outputs (multi-class or sigmoid)
        lp_list = loaded_probes.get(lw, [])
        if lp_list:
            # Take first probe's output for coloring
            p = lp_list[0]
            with torch.no_grad():
                out = p(emb_d.unsqueeze(0))
                if out.dim() == 2 and out.size(1) > 1:
                    probs = F.softmax(out, dim=-1)[0].cpu()
                    pred = int(torch.argmax(out, dim=-1).item())
                    # Map predicted class index to palette
                    palette = ["#a6e3a1", "#fbc1cc", "#c4b5fd", "#fef08a", "#fca5a5", "#ffd1a8"]
                    color = palette[pred % len(palette)]
                else:
                    prob = float(torch.sigmoid(out).item())
                    color = score_to_color(prob)

        # If no linear probe or still no color, use projection similarity
        if color is None and lw in loaded_projs:
            W = loaded_projs[lw]
            z = (emb @ W.T)
            # map mean dimension through tanh into [0,1]
            v = float(torch.tanh(z.mean()).item())
            v = (v + 1.0) / 2.0
            color = score_to_color(v)

        if color is None:
            color = "transparent"

        token_html = f'<span title="{html.escape(tooltip)}" style="background: {color}; padding:2px; border-radius:3px">{html.escape(word)}</span>'
        out_parts.append(token_html)
        last = end

    out_parts.append(html.escape(text[last:]))

    # Add a small legend
    legend_html = """
<div style="display:flex;gap:12px;padding:8px;background:#f5f5f5;border-radius:6px;margin-bottom:10px;">
  <div style="display:flex;flex-direction:column;align-items:center;"><div style="width:30px;height:18px;background:#a6e3a1;border-radius:3px"></div><small>Class 0</small></div>
  <div style="display:flex;flex-direction:column;align-items:center;"><div style="width:30px;height:18px;background:#fbc1cc;border-radius:3px"></div><small>Class 1</small></div>
  <div style="display:flex;flex-direction:column;align-items:center;"><div style="width:30px;height:18px;background:#c4b5fd;border-radius:3px"></div><small>Class 2</small></div>
  <div style="display:flex;flex-direction:column;align-items:center;"><div style="width:30px;height:18px;background:#fef08a;border-radius:3px"></div><small>Medium</small></div>
  <div style="display:flex;flex-direction:column;align-items:center;"><div style="width:30px;height:18px;background:#fca5a5;border-radius:3px"></div><small>High</small></div>
</div>
"""

    html_content = f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>All polysem probes</title></head><body style=\"white-space: pre-wrap; font-family: serif;\">{legend_html}{''.join(out_parts)}</body></html>"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()

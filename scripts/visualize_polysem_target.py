"""
Visualize a target word in a document by coloring each occurrence according to a probe.

Supports two probe types:
 - Linear probe checkpoint (checkpoint contains 'model_state_dict') -> applies probe to token embedding and maps to class or probability.
 - Contrastive projection checkpoint (contains 'W') -> builds centroids from dataset and assigns nearest centroid.

Usage examples:
  python scripts/visualize_polysem_target.py --word bank --text-file alice_in_wonderland.txt --probe checkpoints/polysem_bank_probe.pt --output visualizations/alice_bank.html
  python scripts/visualize_polysem_target.py --word bank --text-file alice_in_wonderland.txt --proj checkpoints/polysem_contrastive_bank_proj.pt --dataset polysemous_datasets.json --output visualizations/alice_bank_contrastive.html

If `--text-file` is not provided, the script will attempt to download Alice from Project Gutenberg.
"""
from pathlib import Path
import argparse
import re
import html
import json
import ssl
import urllib.request
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def download_alice(save_path: Path) -> Path:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(GUTENBERG_URL, context=ctx) as resp:
        data = resp.read()
        text = data.decode("utf-8", errors="replace")
        save_path.write_text(text, encoding="utf-8")
        return save_path


def _word_to_token_index(text: str, word_index: int, tokenizer, max_len: int = 512) -> int | None:
    words = text.split()
    if word_index is None or word_index < 0 or word_index >= len(words):
        return None
    prefix = " ".join(words[: word_index + 1])
    try:
        encoded = tokenizer(prefix, truncation=True, max_length=max_len, return_tensors="pt")
        token_ids = encoded["input_ids"][0]
        if len(token_ids) == 0:
            return None
        return int(len(token_ids) - 1)
    except Exception:
        return None


def extract_embedding_for_occurrence(model, tokenizer, full_text: str, occ_start: int, occ_end: int, max_len=256, use_token_position=True):
    """Extract embedding for the word occurrence by tokenizing a context window and selecting the token index for the occurrence.

    Returns a 1D tensor (hidden_dim,).
    """
    # Build a context window of words around the occurrence
    words = full_text.split()
    # Find which word index corresponds to occ_start by walking chars
    cum = 0
    target_word_idx = None
    for i, w in enumerate(words):
        start = cum
        end = cum + len(w)
        # match approximate by substring — splits may remove punctuation
        if start <= occ_start <= end or (occ_start <= start and end <= occ_end):
            target_word_idx = i
            break
        cum = end + 1

    # Use a window of ±40 words
    left = max(0, (target_word_idx or 0) - 40)
    right = min(len(words), (target_word_idx or 0) + 40)
    context = " ".join(words[left:right])

    inputs = tokenizer(context, truncation=True, max_length=max_len, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1][0]  # (seq, hidden)

    if use_token_position and target_word_idx is not None:
        # map target_word_idx within the context window
        local_word_idx = target_word_idx - left
        token_idx = _word_to_token_index(context, local_word_idx, tokenizer, max_len=max_len)
        if token_idx is None or token_idx < 0 or token_idx >= last_hidden.size(0):
            vec = last_hidden.mean(dim=0)
        else:
            vec = last_hidden[token_idx]
    else:
        vec = last_hidden.mean(dim=0)

    return vec.cpu()


def load_linear_probe(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    # infer output dim from weight if present
    weight = None
    for k, v in state.items():
        if k.endswith("weight"):
            weight = v
            break
    if weight is not None:
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]
    else:
        in_dim = ckpt.get("input_dim")
        out_dim = ckpt.get("num_classes") or 1

    probe = nn.Linear(in_dim, out_dim)
    probe.load_state_dict(state)
    probe = probe.to(device)
    probe.eval()
    meta = {k: ckpt.get(k) for k in ("word", "num_classes", "input_dim")}
    return probe, meta


def load_projection(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    W = ckpt.get("W")
    if isinstance(W, torch.Tensor):
        W = W
    else:
        W = torch.tensor(W)
    return W, ckpt


def generate_color_for_score(score: float):
    # score in [0,1] -> green->yellow->red
    v = max(0.0, min(1.0, float(score)))
    if v < 0.5:
        r = int(255 * (v * 2))
        g = int(100 + 155 * (v * 2))
        b = 0
    else:
        r = 255
        g = int(255 * (1 - (v - 0.5) * 2))
        b = 0
    return f"rgb({r},{g},{b})"


def main():
    parser = argparse.ArgumentParser(description="Visualize target word senses in a document")
    parser.add_argument("--word", type=str, required=True)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--probe", type=str, default=None, help="Path to linear probe checkpoint")
    parser.add_argument("--proj", type=str, default=None, help="Path to contrastive projection checkpoint")
    parser.add_argument("--dataset", type=str, default="polysemous_datasets.json", help="Dataset JSON used to build centroids (for projection)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="visualizations/target_viz.html")
    parser.add_argument("--use-token-position", action="store_true")
    args = parser.parse_args()

    # Load text
    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            print(f"Text file not found: {text_path}")
            return
        text = text_path.read_text(encoding="utf-8")
    else:
        print("Downloading Alice in Wonderland...")
        tpath = Path("alice_in_wonderland.txt")
        text = download_alice(tpath).read_text(encoding="utf-8")

    # Find occurrences (word boundaries, case-insensitive)
    pattern = re.compile(rf"\b{re.escape(args.word)}\b", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        print(f"No occurrences of '{args.word}' found in text")
        return

    device = torch.device(args.device)
    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    results = []  # list of dicts with start,end,pred,score

    if args.probe:
        probe, meta = load_linear_probe(Path(args.probe), device)
        # For each occurrence, extract embedding and run probe
        for m in matches:
            start, end = m.start(), m.end()
            emb = extract_embedding_for_occurrence(model, tokenizer, text, start, end, use_token_position=args.use_token_position)
            emb = emb.to(device)
            with torch.no_grad():
                out = probe(emb.unsqueeze(0))
                if out.dim() == 2 and out.size(1) > 1:
                    probs = F.softmax(out, dim=-1)[0].cpu().tolist()
                    pred = int(torch.argmax(out, dim=-1).item())
                    score = probs
                else:
                    prob = torch.sigmoid(out).item()
                    pred = 1 if prob > 0.5 else 0
                    score = prob
            results.append({"start": start, "end": end, "pred": pred, "score": score})

    elif args.proj:
        W, ckpt = load_projection(Path(args.proj))
        # Need dataset to compute centroids per sense
        try:
            from polysem.polysemous_loader import load_datasets
            datasets = load_datasets(args.dataset)
        except Exception:
            # fallback: try loading directly
            import importlib.util
            repo_root = Path(__file__).resolve().parents[1]
            loader_path = repo_root / "polysem" / "polysemous_loader.py"
            spec = importlib.util.spec_from_file_location("polysemous_loader", str(loader_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            datasets = mod.load_datasets(args.dataset)

        if args.word not in datasets:
            print(f"Word '{args.word}' not found in dataset; cannot compute centroids")
            return

        dataset = datasets[args.word]
        examples = dataset.examples

        # Extract embeddings for dataset examples
        embs = []
        labels = []
        for ex in examples:
            vec = extract_embedding_for_occurrence(model, tokenizer, ex.text, 0, 0, use_token_position=args.use_token_position)
            embs.append(vec)
            labels.append(ex.sense_id)
        embs = torch.stack(embs, dim=0)  # (N, hidden)

        # Project
        Z = embs @ W.T
        # Compute centroids per sense
        centroids = {}
        label_to_vecs = defaultdict(list)
        for z, lab in zip(Z, labels):
            label_to_vecs[lab].append(z)
        for lab, vecs in label_to_vecs.items():
            centroids[lab] = torch.stack(vecs, dim=0).mean(dim=0)

        # For each occurrence, extract embedding, project and find nearest centroid (cosine)
        for m in matches:
            start, end = m.start(), m.end()
            emb = extract_embedding_for_occurrence(model, tokenizer, text, start, end, use_token_position=args.use_token_position)
            z = emb @ W.T
            best_lab = None
            best_sim = -1e9
            for lab, c in centroids.items():
                sim = F.cosine_similarity(z.unsqueeze(0), c.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_lab = lab
            results.append({"start": start, "end": end, "pred": int(best_lab), "score": float(best_sim)})

    else:
        print("ERROR: must provide --probe or --proj")
        return

    # Build HTML by replacing occurrences with spans
    out_parts = []
    last = 0
    for i, m in enumerate(matches):
        start, end = m.start(), m.end()
        out_parts.append(html.escape(text[last:start]))
        res = results[i]
        if isinstance(res["score"], list):
            # multi-class probabilities -> color by predicted class index
            pred = res["pred"]
            palette = ["#a6e3a1", "#fbc1cc", "#c4b5fd", "#fef08a", "#fca5a5"]
            color = palette[pred % len(palette)]
        elif isinstance(res["score"], float):
            if args.probe and isinstance(res["score"], float) and (not isinstance(res["score"], list)):
                color = generate_color_for_score(res["score"])
            else:
                # projection similarity
                v = (res["score"] + 1) / 2  # map cosine [-1,1] to [0,1]
                color = generate_color_for_score(v)
        else:
            color = "transparent"

        token_html = f'<span class="target" style="background:{color};padding:2px;border-radius:3px;">{html.escape(text[start:end])}</span>'
        out_parts.append(token_html)
        last = end
    out_parts.append(html.escape(text[last:]))

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Target visualization: {args.word}</title></head>
<body style="font-family: serif; white-space: pre-wrap;">{''.join(out_parts)}</body></html>"""

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    import argparse
    main()

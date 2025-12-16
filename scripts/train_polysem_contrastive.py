"""
Train a supervised-contrastive linear probe for word-sense disambiguation.

Architecture:
  projection(x) = x @ W  where W is (hidden_dim, output_dim)

Loss: Supervised Contrastive Loss (Khosla et al.) implemented in-batch.

Usage example:
  python scripts/train_polysem_contrastive.py --word bank --dataset polysemous_datasets.json \
      --model gpt2 --device cuda --epochs 50 --batch-size 16 --output-dim 64 --temperature 0.1

Saves checkpoint to `--probes-out` directory containing projection `W`.
"""
from pathlib import Path
import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_load_datasets():
    try:
        from polysem.polysemous_loader import load_datasets
        return load_datasets
    except Exception:
        import importlib.util
        repo_root = Path(__file__).resolve().parents[1]
        loader_path = repo_root / "polysem" / "polysemous_loader.py"
        spec = importlib.util.spec_from_file_location("polysemous_loader", str(loader_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Train supervised-contrastive linear probe for polysemous word senses")
    p.add_argument("--word", type=str, required=True)
    p.add_argument("--dataset", type=str, default="polysemous_datasets.json")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--probes-out", type=str, default="checkpoints/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-token-position", action="store_true", help="Use token-position embedding of target word (fallback to mean)")
    return p.parse_args()


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


def extract_embeddings(model, tokenizer, examples, device, max_len=256, use_token_position=False):
    """Return tensor of shape (n_examples, hidden_dim).

    If use_token_position, `examples` are SenseExample objects with `text` and `find_word_position()`.
    Otherwise examples is list[str].
    """
    model.eval()
    embs = []
    with torch.no_grad():
        if use_token_position:
            for ex in examples:
                text = ex.text
                word_idx = ex.find_word_position()
                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]  # (seq, hidden_dim)
                token_idx = _word_to_token_index(text, word_idx, tokenizer, max_len=max_len)
                if token_idx is None or token_idx < 0 or token_idx >= last_hidden.size(0):
                    vec = last_hidden.mean(dim=0)
                else:
                    vec = last_hidden[token_idx]
                embs.append(vec.cpu())
        else:
            for item in examples:
                if isinstance(item, str):
                    text = item
                elif hasattr(item, "text"):
                    text = item.text
                else:
                    text = str(item)

                inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1][0]
                vec = last_hidden.mean(dim=0)
                embs.append(vec.cpu())

    return torch.stack(embs, dim=0)


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float):
    """Compute supervised contrastive loss for a batch.

    z: (B, D) assumed L2-normalized
    labels: (B,) integer labels
    returns scalar loss
    """
    device = z.device
    B = z.size(0)
    sim = torch.matmul(z, z.T) / temperature  # (B, B)
    # mask out self-similarities
    logits_mask = (~torch.eye(B, dtype=torch.bool, device=device)).float()

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # positives mask (B,B)

    # For numerical stability, subtract max per row
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    exp_sim = torch.exp(sim) * logits_mask
    # denominator: sum_{k != i} exp(sim(i,k))
    denom = exp_sim.sum(dim=1, keepdim=True)  # (B,1)

    # numerator: sum over positives (exclude self)
    positive_mask = mask * logits_mask
    numerator = (exp_sim * positive_mask).sum(dim=1)

    # For anchors with no positives (shouldn't happen if batch contains at least one other same-label), avoid NaN
    eps = 1e-12
    loss_per_sample = -torch.log((numerator + eps) / (denom + eps))

    # Only average over samples that have at least one positive
    valid = (positive_mask.sum(dim=1) > 0).float()
    loss = (loss_per_sample * valid).sum() / (valid.sum() + 1e-12)
    return loss


def build_stratified_batch_indices(labels: torch.Tensor, batch_size: int, min_pos: int = 2):
    """Yield index tensors for stratified batches.

    Strategy: pick some number of classes per batch, and sample up to `min_pos`
    examples per selected class to fill the batch. If dataset small, fall back
    to random sampling while ensuring at least two classes when possible.
    """
    device = labels.device
    N = labels.size(0)
    labels_cpu = labels.cpu().numpy()
    unique = list(set(labels_cpu.tolist()))
    # Build mapping class -> indices
    from collections import defaultdict
    cls_to_idx = defaultdict(list)
    for i, l in enumerate(labels_cpu.tolist()):
        cls_to_idx[int(l)].append(i)

    indices = list(range(N))
    random.shuffle(indices)

    batches = []
    i = 0
    while i < N:
        batch = []
        # Attempt to sample classes and examples per class
        classes = random.sample(unique, k=min(len(unique), max(1, batch_size // min_pos)))
        for c in classes:
            avail = cls_to_idx[c]
            k = min(min_pos, len(avail), batch_size - len(batch))
            if k <= 0:
                continue
            sampled = random.sample(avail, k)
            for s in sampled:
                if s not in batch:
                    batch.append(s)
            if len(batch) >= batch_size:
                break

        # Fill remaining slots with random unused indices
        if len(batch) < batch_size:
            for idx in indices:
                if idx not in batch:
                    batch.append(idx)
                if len(batch) >= batch_size:
                    break

        batches.append(torch.tensor(batch, dtype=torch.long, device=device))
        i += len(batch)

    return batches


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_datasets = import_load_datasets()
    try:
        datasets = load_datasets(args.dataset)
    except FileNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [Path(args.dataset), repo_root / args.dataset, repo_root / "data" / args.dataset]
        print("ERROR: dataset file not found. Tried:")
        for c in candidates:
            print(f"  - {c}")
        return

    if args.word not in datasets:
        print(f"ERROR: word '{args.word}' not found; available: {list(datasets.keys())}")
        return

    dataset = datasets[args.word]
    examples = dataset.examples
    labels = torch.tensor([ex.sense_id for ex in examples], dtype=torch.long)

    # Load model and compute embeddings
    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    print("Extracting embeddings...")
    embs = extract_embeddings(model, tokenizer, examples, args.device, max_len=args.max_len, use_token_position=args.use_token_position)
    # embs: (N, hidden_dim) on CPU

    N, hidden_dim = embs.size()
    print(f"Extracted {N} embeddings, dim={hidden_dim}")

    # Move embeddings and labels to device for training
    embs = embs.to(args.device)
    labels = labels.to(args.device)

    # Projection (probe)
    proj = nn.Linear(hidden_dim, args.output_dim, bias=False).to(args.device)
    # Initialize with small weights
    nn.init.normal_(proj.weight, mean=0.0, std=0.01)

    opt = torch.optim.Adam(proj.parameters(), lr=args.lr)

    # Dataset for sampling indices
    dataset_inds = torch.arange(N)

    for epoch in range(1, args.epochs + 1):
        proj.train()
        total_loss = 0.0
        count = 0

        # Build stratified batches per epoch
        batches = build_stratified_batch_indices(labels.cpu(), args.batch_size, min_pos=2)

        for batch_idx in batches:
            batch_embs = embs[batch_idx]
            batch_labels = labels[batch_idx]

            # Ensure at least two unique labels in batch
            if torch.unique(batch_labels).numel() <= 1:
                continue

            z = proj(batch_embs)  # (b, out)
            z = F.normalize(z, dim=-1)

            loss = supervised_contrastive_loss(z, batch_labels, args.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * batch_idx.size(0)
            count += batch_idx.size(0)

        avg_loss = total_loss / max(1, count)
        print(f"Epoch {epoch}/{args.epochs} — avg_loss: {avg_loss:.4f}")

    # Save projection matrix
    out_dir = Path(args.probes_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polysem_contrastive_{args.word}_proj.pt"
    checkpoint = {
        "W": proj.weight.detach().cpu(),
        "input_dim": hidden_dim,
        "output_dim": args.output_dim,
        "temperature": args.temperature,
        "word": args.word,
    }
    torch.save(checkpoint, str(out_path))
    print(f"Saved projection checkpoint to: {out_path}")


if __name__ == "__main__":
    main()

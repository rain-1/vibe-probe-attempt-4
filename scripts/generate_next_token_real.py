"""
Data generation script for next-token probe training using REAL TEXT.

Uses text from books (Project Gutenberg) instead of model-generated text
to get natural, diverse patterns.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate probe training data from real text"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Target number of samples (balanced, default: 2000)",
    )
    parser.add_argument(
        "--target-token",
        type=str,
        default=" the",
        help="Target token to predict (default: ' the')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/next_token_real.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-270m",
        help="Model to use for hidden states",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=64,
        help="Context length in tokens (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use left-padding for causal LMs - critical for correct batched extraction!
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_book_text():
    """Load book text from cached files."""
    cache_dir = Path("data/books")
    combined = ""
    
    for txt_file in cache_dir.glob("*.txt"):
        print(f"Loading {txt_file.name}...")
        combined += txt_file.read_text(encoding="utf-8") + "\n\n"
    
    if not combined:
        raise ValueError("No book files found in data/books/")
    
    return combined


def extract_hidden_states_batch(model, tokenizer, texts: list[str], device: str, batch_size: int = 32) -> list[dict]:
    """Extract hidden states from all layers for a batch of texts.
    
    Uses left-padding so the last position is always the real last token.
    """
    all_results = []
    
    # Get final norm
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[batch_start:batch_start + batch_size]
        
        # With left-padding, last position is always the real last token
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        hidden_states = outputs.hidden_states
        
        for i in range(len(batch_texts)):
            result = {}
            
            # With left-padding, always use position -1 (last position)
            last_pos = -1
            
            for layer_idx, layer_hidden in enumerate(hidden_states):
                last_hidden = layer_hidden[i, last_pos, :].cpu().float().numpy().tolist()
                result[f"layer-{layer_idx}-hidden"] = last_hidden
            
            if final_norm is not None:
                last_layer_hidden = hidden_states[-1][i, last_pos, :].unsqueeze(0)
                normed_hidden = final_norm(last_layer_hidden)
                result["layer-final-hidden"] = normed_hidden.squeeze(0).detach().cpu().float().numpy().tolist()
            
            all_results.append(result)
    
    return all_results


def verify_extraction(model, tokenizer, texts: list[str], hidden_states: list[dict], device: str, n_samples: int = 10):
    """Verify batched extraction matches single-sample extraction.
    
    Compares a sample of batched hidden states against single-sample ground truth.
    Returns True if extraction is correct (cosine similarity > 0.999).
    """
    import numpy as np
    
    print(f"\nVerifying extraction quality on {n_samples} samples...")
    
    final_norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
    
    cos_sims = []
    lm_head_matches = []
    
    indices = random.sample(range(len(texts)), min(n_samples, len(texts)))
    
    for idx in indices:
        text = texts[idx]
        stored = np.array(hidden_states[idx]["layer-final-hidden"], dtype=np.float32)
        
        # Single-sample extraction (ground truth)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            single_hidden = outputs.hidden_states[-1][0, -1]
            if final_norm is not None:
                single_hidden = final_norm(single_hidden.unsqueeze(0)).squeeze()
            single_hidden = single_hidden.float().cpu().numpy()
        
        # Cosine similarity
        cos_sim = np.dot(stored, single_hidden) / (np.linalg.norm(stored) * np.linalg.norm(single_hidden))
        cos_sims.append(cos_sim)
        
        # lm_head prediction match
        with torch.no_grad():
            model_dtype = next(model.parameters()).dtype
            stored_logits = model.lm_head(torch.tensor(stored).unsqueeze(0).to(device).to(model_dtype))
            single_logits = model.lm_head(torch.tensor(single_hidden).unsqueeze(0).to(device).to(model_dtype))
            stored_pred = stored_logits.argmax(dim=-1).item()
            single_pred = single_logits.argmax(dim=-1).item()
            lm_head_matches.append(stored_pred == single_pred)
    
    avg_cos_sim = np.mean(cos_sims)
    match_rate = np.mean(lm_head_matches)
    
    print(f"  Average cosine similarity: {avg_cos_sim:.6f}")
    print(f"  lm_head prediction match rate: {match_rate:.1%}")
    
    if avg_cos_sim < 0.999:
        print("  WARNING: Low cosine similarity! Batched extraction may be corrupted.")
        return False
    else:
        print("  OK: Extraction verified.")
        return True


def main():
    args = parse_args()
    random.seed(args.seed)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Get target token ID
    target_ids = tokenizer.encode(args.target_token, add_special_tokens=False)
    if len(target_ids) != 1:
        print(f"Warning: '{args.target_token}' -> {len(target_ids)} tokens")
    target_id = target_ids[0]
    print(f"Target: '{args.target_token}' -> ID {target_id}")
    
    # Load book text
    text = get_book_text()
    print(f"Loaded {len(text):,} characters")
    
    # Tokenize
    print("Tokenizing...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(tokens):,}")
    
    # Find all positions where target token appears
    print("Finding samples...")
    positive_indices = []  # Indices where tokens[i] == target
    negative_indices = []
    
    for i in range(args.context_len, len(tokens)):
        if tokens[i] == target_id:
            positive_indices.append(i)
        else:
            negative_indices.append(i)
    
    print(f"Positive (next=target): {len(positive_indices):,}")
    print(f"Negative (next!=target): {len(negative_indices):,}")
    
    # Balance and sample
    k = min(len(positive_indices), len(negative_indices), args.n // 2)
    print(f"Sampling {k} from each class...")
    
    selected_pos = random.sample(positive_indices, k)
    selected_neg = random.sample(negative_indices, k)
    
    # Create samples - context is tokens BEFORE the target position
    samples = []
    
    for idx in selected_pos:
        # Context ends at idx-1, next token is at idx (which is target)
        context_tokens = tokens[idx - args.context_len:idx]
        context_text = tokenizer.decode(context_tokens)
        samples.append({
            "text": context_text,
            "label": 1,
            "next_token_id": tokens[idx],
            "next_token_text": tokenizer.decode([tokens[idx]]),
        })
    
    for idx in selected_neg:
        context_tokens = tokens[idx - args.context_len:idx]
        context_text = tokenizer.decode(context_tokens)
        samples.append({
            "text": context_text,
            "label": 0,
            "next_token_id": tokens[idx],
            "next_token_text": tokenizer.decode([tokens[idx]]),
        })
    
    random.shuffle(samples)
    
    # Extract hidden states
    print(f"\nExtracting hidden states for {len(samples)} samples...")
    all_texts = [s["text"] for s in samples]
    hidden_states = extract_hidden_states_batch(model, tokenizer, all_texts, args.device, args.batch_size)
    
    # Verify extraction quality
    verify_extraction(model, tokenizer, all_texts, hidden_states, args.device, n_samples=20)
    
    # Merge
    final_rows = []
    for i, sample in enumerate(samples):
        row = sample.copy()
        row.update(hidden_states[i])
        final_rows.append(row)
    
    df = pd.DataFrame(final_rows)
    df.to_parquet(output_path, index=False)
    
    print(f"\nSaved {len(df)} samples to {output_path}")
    print(f"  Positive: {sum(df['label'] == 1)}")
    print(f"  Negative: {sum(df['label'] == 0)}")
    
    # Show some examples
    print("\nSample positive contexts (before ' the'):")
    for _, r in df[df.label == 1].head(5).iterrows():
        print(f"  ...{r.text[-40:]}")


if __name__ == "__main__":
    main()

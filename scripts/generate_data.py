"""
Data generation script for probe training.

Generates synthetic text data with hidden layer activations for training
a linear probe to predict target tokens.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate probe training data with hidden layer activations"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--target-token",
        type=str,
        default=" the",
        help="Target token T for positive/negative split (default: ' the')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/probe_data.parquet",
        help="Output parquet file path (default: data/probe_data.parquet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-270m",
        help="Model to use (default: unsloth/gemma-3-270m)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--warm-tokens",
        type=int,
        default=20,
        help="Number of warm-up tokens to generate at temp=1.0 (default: 20)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate at temp=0 (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel generation (default: 1)",
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


def generate_text_samples_batch(model, tokenizer, device: str, warm_tokens: int, gen_tokens: int, batch_size: int) -> list[tuple[str, torch.Tensor]]:
    """
    Generate a batch of text samples in parallel:
    1. Generate warm_tokens at temp=1.0 (diversity) for all sequences
    2. Continue with gen_tokens at temp=0 (deterministic) for all sequences
    3. Return final gen_tokens only for each
    
    Returns:
        List of (decoded text, token tensor) tuples
    """
    # Start with batch of prompts
    input_ids = tokenizer.encode("The", return_tensors="pt").to(device)
    input_ids = input_ids.repeat(batch_size, 1)  # Expand to batch
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate warm_tokens at temp=1.0 with top_k/top_p for safe sampling
    with torch.no_grad():
        output_high_temp = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=warm_tokens,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Update attention mask for generated tokens
    attention_mask = torch.ones_like(output_high_temp)
    
    # Continue with gen_tokens at temp=0 (greedy)
    with torch.no_grad():
        output_full = model.generate(
            output_high_temp,
            attention_mask=attention_mask,
            max_new_tokens=gen_tokens,
            temperature=None,  # Greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract final gen_tokens for each sequence in batch
    results = []
    for i in range(batch_size):
        final_tokens = output_full[i, -gen_tokens:]
        text = tokenizer.decode(final_tokens, skip_special_tokens=True)
        results.append((text, final_tokens))
    
    return results


def extract_hidden_states_batch(model, tokenizer, texts: list[str], device: str, target_token_id: int, batch_size: int = 32) -> list[dict]:
    """
    Extract hidden states from all layers for a batch of texts.
    Returns a list of dicts with layer-N-hidden keys containing the final token's hidden state,
    plus the logit value and probability for the target token.
    """
    all_results = []
    
    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        
        # Tokenize with padding
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden_dim)
        logits = outputs.logits  # (batch, seq, vocab_size)
        attention_mask = inputs.attention_mask  # (batch, seq)
        
        # Process each sample in batch
        for i in range(len(batch_texts)):
            result = {}
            
            # Find last non-padded position for this sample
            seq_len = attention_mask[i].sum().item()
            last_pos = seq_len - 1
            
            # Extract hidden states from each layer at last position
            for layer_idx, layer_hidden in enumerate(hidden_states):
                last_hidden = layer_hidden[i, last_pos, :].cpu().float().numpy().tolist()
                result[f"layer-{layer_idx}-hidden"] = last_hidden
            
            # Get logits for the last position
            last_logits = logits[i, last_pos, :]  # (vocab_size,)
            
            # Raw logit for target token
            target_logit = last_logits[target_token_id].cpu().float().item()
            result["target_token_logit"] = target_logit
            
            # Probability via softmax (interpretable 0-1 range)
            probs = torch.softmax(last_logits, dim=0)
            target_prob = probs[target_token_id].cpu().float().item()
            result["target_token_prob"] = target_prob
            
            all_results.append(result)
    
    return all_results


def main():
    args = parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Get target token ID
    target_token = args.target_token
    # Encode the target token to get its ID
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_token_ids) != 1:
        print(f"WARNING: Target token '{target_token}' encodes to {len(target_token_ids)} tokens: {target_token_ids}")
        print("Using the last token ID for matching.")
    target_token_id = target_token_ids[-1]
    print(f"Target token: '{target_token}' -> token ID: {target_token_id}")
    
    # Generate samples and collect ALL substrings ending with target token
    print(f"Generating {args.n} text samples (batch_size={args.batch_size})...")
    positive_positions = []  # List of (tokens_tensor, end_position) where tokens[end_pos] == target
    negative_positions = []  # List of (tokens_tensor, end_position) where tokens[end_pos] != target
    
    # Calculate number of batches
    num_batches = (args.n + args.batch_size - 1) // args.batch_size
    samples_generated = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate actual batch size for this iteration (last batch may be smaller)
        current_batch_size = min(args.batch_size, args.n - samples_generated)
        
        batch_results = generate_text_samples_batch(
            model, tokenizer, args.device, args.warm_tokens, args.gen_tokens, current_batch_size
        )
        
        for text, tokens in batch_results:
            # Find all positions in this sequence (require at least 8 tokens of context)
            for pos in range(8, len(tokens)):
                token_id = tokens[pos].item()
                if token_id == target_token_id:
                    positive_positions.append((tokens, pos))
                else:
                    negative_positions.append((tokens, pos))
        
        samples_generated += current_batch_size
    
    print(f"Positive substrings (end with token {target_token_id}): {len(positive_positions)}")
    print(f"Negative substrings: {len(negative_positions)}")
    
    # Balance the dataset
    k = min(len(positive_positions), len(negative_positions))
    if k == 0:
        print("ERROR: No positive or negative samples found. Try increasing --n or changing --target-token")
        return
    
    # Shuffle and take k of each
    import random
    random.shuffle(positive_positions)
    random.shuffle(negative_positions)
    positive_positions = positive_positions[:k]
    negative_positions = negative_positions[:k]
    
    print(f"Balanced dataset size: {k} positive + {k} negative = {2*k} total")
    
    # Extract hidden states for all samples (batched)
    print("Extracting hidden states (batched)...")
    
    # Prepare positive samples
    positive_texts = []
    positive_metadata = []  # (text, next_token_id)
    for tokens, end_pos in positive_positions:
        tokens_without_final = tokens[:end_pos]
        text_prefix = tokenizer.decode(tokens_without_final, skip_special_tokens=True)
        positive_texts.append(text_prefix)
        positive_metadata.append((text_prefix, target_token_id))
    
    # Prepare negative samples
    negative_texts = []
    negative_metadata = []  # (text, next_token_id)
    for tokens, end_pos in negative_positions:
        tokens_without_final = tokens[:end_pos]
        text_prefix = tokenizer.decode(tokens_without_final, skip_special_tokens=True)
        negative_texts.append(text_prefix)
        negative_metadata.append((text_prefix, tokens[end_pos].item()))
    
    # Extract hidden states in batches
    print(f"Processing {len(positive_texts)} positive samples...")
    positive_hidden = extract_hidden_states_batch(
        model, tokenizer, positive_texts, args.device, target_token_id, batch_size=args.batch_size
    )
    
    print(f"Processing {len(negative_texts)} negative samples...")
    negative_hidden = extract_hidden_states_batch(
        model, tokenizer, negative_texts, args.device, target_token_id, batch_size=args.batch_size
    )
    
    # Build data rows
    data_rows = []
    for i, hidden_states in enumerate(positive_hidden):
        text, next_token_id = positive_metadata[i]
        row = {
            "text": text,
            "label": 1,
            "next_token_id": next_token_id,
            **hidden_states,
        }
        data_rows.append(row)
    
    for i, hidden_states in enumerate(negative_hidden):
        text, next_token_id = negative_metadata[i]
        row = {
            "text": text,
            "label": 0,
            "next_token_id": next_token_id,
            **hidden_states,
        }
        data_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    df.to_parquet(output_path, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample hidden state shape
    first_hidden_col = [c for c in df.columns if c.startswith("layer-")][0]
    hidden_dim = len(df[first_hidden_col].iloc[0])
    num_layers = len([c for c in df.columns if c.startswith("layer-")])
    print(f"Hidden dimensions: {hidden_dim}")
    print(f"Number of layers: {num_layers}")


if __name__ == "__main__":
    main()

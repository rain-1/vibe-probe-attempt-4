"""
Data generation script for next-token probe training.

Generates training data for a probe to predict if a specific target token
will be the next token, using model-generated text.
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate probe training data (next token prediction)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Target number of samples to generate (will be balanced, default: 2000)",
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
        default="data/next_token_data.parquet",
        help="Output parquet file path (default: data/next_token_data.parquet)",
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
        "--context-len",
        type=int,
        default=64,
        help="Context length for each sample in tokens (default: 64)",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=100,
        help="Number of warmup tokens at temp=1.0 (default: 100)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=1000,
        help="Number of tokens to generate at low temp (default: 1000)",
    )
    parser.add_argument(
        "--gen-temp",
        type=float,
        default=0.0,
        help="Temperature for main generation (default: 0.0)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=50,
        help="Number of separate generation runs (default: 50)",
    )
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=32,
        help="Batch size for hidden state extraction (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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


def generate_text_batch(model, tokenizer, device: str, warmup_tokens: int, gen_tokens: int, 
                        gen_temp: float, num_generations: int) -> list[list[int]]:
    """
    Generate multiple sequences of text.
    Uses temp=1.0 warmup, then low temp for main generation.
    Returns list of token id lists (excluding warmup).
    """
    all_tokens = []
    
    for i in tqdm(range(num_generations), desc="Generating text"):
        # Start with a simple prompt
        prompt = "The"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Warmup at temp=1.0
            warmup_output = model.generate(
                input_ids,
                max_new_tokens=warmup_tokens,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Continue at low temp
            if gen_temp == 0.0:
                # Greedy decoding
                output = model.generate(
                    warmup_output,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                output = model.generate(
                    warmup_output,
                    max_new_tokens=gen_tokens,
                    temperature=gen_temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        # Extract only the low-temp tokens (skip prompt + warmup)
        prompt_len = input_ids.shape[1]
        low_temp_tokens = output[0, prompt_len + warmup_tokens:].tolist()
        all_tokens.append(low_temp_tokens)
    
    return all_tokens


def extract_hidden_states_batch(model, tokenizer, texts: list[str], device: str, batch_size: int = 32) -> list[dict]:
    """
    Extract hidden states from all layers for a batch of texts.
    Returns a list of dicts with layer-N-hidden keys containing the final token's hidden state.
    Also extracts post-final-layer-norm hidden states as 'layer-final-hidden'.
    """
    all_results = []
    
    # Get the final layer norm (works for Gemma, Llama, etc.)
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm  # Gemma, Llama style
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        final_norm = model.transformer.ln_f  # GPT-2 style
    
    if final_norm is None:
        print("Warning: Could not find final layer norm. 'layer-final-hidden' will not be extracted.")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
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
            
            # Extract post-final-layer-norm hidden state
            if final_norm is not None:
                last_layer_hidden = hidden_states[-1][i, last_pos, :].unsqueeze(0)
                normed_hidden = final_norm(last_layer_hidden)
                result["layer-final-hidden"] = normed_hidden.squeeze(0).detach().cpu().float().numpy().tolist()
            
            all_results.append(result)
    
    return all_results


def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Get target token ID
    target_token_ids = tokenizer.encode(args.target_token, add_special_tokens=False)
    if len(target_token_ids) != 1:
        print(f"Warning: Target '{args.target_token}' tokenizes to {len(target_token_ids)} tokens: {target_token_ids}")
        print("Using first token only.")
    target_token_id = target_token_ids[0]
    print(f"Target token: '{args.target_token}' -> ID {target_token_id}")
    
    # Generate text
    print(f"\nGenerating {args.num_generations} sequences...")
    all_token_lists = generate_text_batch(
        model, tokenizer, args.device,
        args.warmup_tokens, args.gen_tokens, args.gen_temp, args.num_generations
    )
    
    # Flatten and create samples
    # Each sample is (context tokens, next token, label)
    print("\nCreating samples...")
    positive_samples = []  # Next token IS target
    negative_samples = []  # Next token is NOT target
    
    for token_list in all_token_lists:
        # For each position where we have enough context
        for i in range(args.context_len, len(token_list)):
            context_tokens = token_list[i - args.context_len:i]
            next_token = token_list[i]
            
            context_text = tokenizer.decode(context_tokens)
            
            if next_token == target_token_id:
                positive_samples.append({
                    "text": context_text,
                    "label": 1,
                    "next_token_id": next_token,
                    "next_token_text": tokenizer.decode([next_token]),
                })
            else:
                negative_samples.append({
                    "text": context_text,
                    "label": 0,
                    "next_token_id": next_token,
                    "next_token_text": tokenizer.decode([next_token]),
                })
    
    print(f"Positive samples (next='{args.target_token}'): {len(positive_samples)}")
    print(f"Negative samples (next!='{args.target_token}'): {len(negative_samples)}")
    
    # Balance the dataset
    k = min(len(positive_samples), len(negative_samples), args.n // 2)
    print(f"Sampling {k} from each class...")
    
    selected_positive = random.sample(positive_samples, k)
    selected_negative = random.sample(negative_samples, k)
    
    samples = selected_positive + selected_negative
    random.shuffle(samples)
    
    # Extract hidden states
    print(f"\nExtracting hidden states for {len(samples)} samples...")
    all_texts = [s["text"] for s in samples]
    hidden_states = extract_hidden_states_batch(
        model, tokenizer, all_texts, args.device, batch_size=args.extract_batch_size
    )
    
    # Merge
    final_rows = []
    for i, sample in enumerate(samples):
        row = sample.copy()
        row.update(hidden_states[i])
        final_rows.append(row)
    
    df = pd.DataFrame(final_rows)
    df.to_parquet(output_path, index=False)
    
    print(f"\nSaved {len(df)} samples to {output_path}")
    print(f"  - Positive: {sum(df['label'] == 1)}")
    print(f"  - Negative: {sum(df['label'] == 0)}")


if __name__ == "__main__":
    main()

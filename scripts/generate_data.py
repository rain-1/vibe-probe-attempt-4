"""
Data generation script for probe training.

Generates training data for a probe to detect if a token is inside a quotation
using text from Project Gutenberg.
"""

import argparse
import os
import re
import requests
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate probe training data (quote detection)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of samples to generate (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/quote_probe_data.parquet",
        help="Output parquet file path (default: data/quote_probe_data.parquet)",
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


def get_book_texts():
    """Download multiple books from Project Gutenberg to ensure diversity."""
    books = {
        "alice": "https://www.gutenberg.org/files/11/11-0.txt",
        "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
        "pride": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt"
    }
    
    combined_text = ""
    cache_dir = Path("data/books")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in books.items():
        cache_path = cache_dir / f"{name}.txt"
        
        if cache_path.exists():
            print(f"Loading cached {name}...")
            text = cache_path.read_text(encoding="utf-8")
        else:
            print(f"Downloading {name} from {url}...")
            try:
                response = requests.get(url)
                response.encoding = "utf-8"
                text = response.text
                
                # Basic cleanup
                start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
                end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
                
                start_idx = text.find(start_marker)
                if start_idx != -1:
                    # Find the end of the line containing the marker
                    newline_idx = text.find("\n", start_idx)
                    if newline_idx != -1:
                        text = text[newline_idx+1:]
                    else: 
                        text = text[start_idx + len(start_marker):]
                    
                end_idx = text.find(end_marker)
                if end_idx != -1:
                    text = text[:end_idx]
                    
                cache_path.write_text(text, encoding="utf-8")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
        
        combined_text += text + "\n\n"
        
    return combined_text


def extract_hidden_states_batch(model, tokenizer, texts: list[str], device: str, batch_size: int = 32) -> list[dict]:
    """
    Extract hidden states from all layers for a batch of texts.
    Returns a list of dicts with layer-N-hidden keys containing the final token's hidden state.
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
            
            all_results.append(result)
    
    return all_results


def main():
    args = parse_args()
    
    # Set seed
    import random
    random.seed(args.seed)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and text
    model, tokenizer = load_model(args.model, args.device)
    text = get_book_texts()
    
    print("Tokenizing entire text...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(tokens)}")
    
    # Label tokens (inside/outside quotes)
    # We'll map tokens back to text to check for quotes
    # This is an approximation. A more robust way would be character-level tracking mapped to tokens.
    
    print("Labeling tokens...")
    # Char level tagging
    # 1. Create a boolean mask for all characters
    # Heuristic:
    # " -> Toggle
    # “ -> Open (True)
    # ” -> Close (True then False)
    
    char_inside = [False] * len(text)
    curr_state = False
    
    for i, char in enumerate(text):
        if char == '“':
            curr_state = True
            char_inside[i] = True # Open quote is inside
        elif char == '”':
            char_inside[i] = True # Closing quote is inside
            curr_state = False    # Then split
        elif char == '"':
            # Ambiguous quote
            if not curr_state:
                # Opening
                curr_state = True
                char_inside[i] = True
            else:
                # Closing
                char_inside[i] = True
                curr_state = False
        else:
            char_inside[i] = curr_state
            
    # 2. Align tokens
    print("Aligning tokens to character labels...")
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.offset_mapping 
    
    token_labels = []
    for (start, end) in offsets:
        segment_labels = char_inside[start:end]
        if not segment_labels:
            token_labels.append(0)
            continue
        
        # If any part is inside, be careful. 
        # Ideally majority.
        avg_inside = sum(segment_labels) / len(segment_labels)
        label = 1 if avg_inside > 0.5 else 0
        token_labels.append(label)
    
    token_labels = torch.tensor(token_labels)
    # Truncate if mismatch (shouldn't happen with correct usage)
    if len(token_labels) != len(tokens):
        print(f"Warning: size mismatch {len(token_labels)} vs {len(tokens)}")
        min_len = min(len(token_labels), len(tokens))
        token_labels = token_labels[:min_len]
        tokens = tokens[:min_len]
    
    
    # Sampling
    # We want contexts of length L ending at some position i.
    # Label is label[i]
    
    # Find indices for 0 and 1
    # Valid indices must be >= context_len
    valid_indices = torch.where(torch.tensor(range(len(tokens))) >= args.context_len)[0]
    
    # Filter by label
    indices_0 = [i.item() for i in valid_indices if token_labels[i] == 0]
    indices_1 = [i.item() for i in valid_indices if token_labels[i] == 1]
    
    print(f"Total available samples: Outside={len(indices_0)}, Inside={len(indices_1)}")
    
    k = min(len(indices_0), len(indices_1), args.n // 2)
    print(f"Sampling {k} from each class...")
    
    import random
    selected_0 = random.sample(indices_0, k)
    selected_1 = random.sample(indices_1, k)
    
    # Create dataset samples
    samples = [] # (text_context, label)
    
    # Helper to get text context
    def get_context(idx):
        # Slice tokens
        start = idx - args.context_len + 1
        end = idx + 1
        ctx_tokens = tokens[start:end]
        return tokenizer.decode(ctx_tokens)
        
    for idx in selected_0:
        samples.append({
            "text": get_context(idx),
            "label": 0
        })
        
    for idx in selected_1:
        samples.append({
            "text": get_context(idx),
            "label": 1
        })
        
    random.shuffle(samples)
    
    # Extract hidden states
    print("Extracting hidden states...")
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
    
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    main()

"""
Comparing embedding extraction approaches for word sense disambiguation:

1. GPT-2 Basic: Embedding at target word position (only past context)
2. GPT-2 Attention Flow: What information about target word reached end of sentence
3. BERT: Embedding at target word position (full bidirectional context)

The key insight: In autoregressive models, disambiguation happens AFTER the ambiguous word.
So we extract what "flowed" from the target word to the final position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


# ============================================================
# Data Loading (same as before)
# ============================================================

@dataclass
class SenseExample:
    text: str
    word: str
    sense: str
    sense_id: int


@dataclass
class PolysemousDataset:
    word: str
    senses: List[str]
    examples: List[SenseExample]
    
    def summary(self) -> str:
        lines = [f"Word: {self.word}", f"Senses: {self.senses}"]
        for sense in self.senses:
            count = len([e for e in self.examples if e.sense == sense])
            lines.append(f"  {sense}: {count}")
        return "\n".join(lines)


def load_datasets(path: str = "polysemous_datasets.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    
    datasets = {}
    for word, word_data in data.items():
        senses = word_data["senses"]
        sense_to_id = {sense: i for i, sense in enumerate(senses)}
        examples = [
            SenseExample(
                text=sent["text"],
                word=word,
                sense=sent["sense"],
                sense_id=sense_to_id[sent["sense"]]
            )
            for sent in word_data["sentences"]
        ]
        datasets[word] = PolysemousDataset(word=word, senses=senses, examples=examples)
    
    return datasets


# ============================================================
# Token Finding Utility
# ============================================================

def find_target_word_tokens(sentence: str, target_word: str, tokenizer) -> List[int]:
    """Find token indices corresponding to the target word."""
    
    # Tokenize with offset mapping
    inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
    offsets = inputs["offset_mapping"][0].tolist()
    
    word_lower = target_word.lower()
    sentence_lower = sentence.lower()
    
    # Find character position of target word
    start_char = sentence_lower.find(word_lower)
    if start_char == -1:
        for suffix in ['s', 'es', 'ed', 'ing']:
            start_char = sentence_lower.find(word_lower + suffix)
            if start_char != -1:
                end_char = start_char + len(word_lower) + len(suffix)
                break
        else:
            return []
    else:
        end_char = start_char + len(word_lower)
    
    # Find tokens that overlap with target word
    token_positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end == 0:  # Skip special tokens
            continue
        if tok_start < end_char and tok_end > start_char:
            token_positions.append(idx)
    
    return token_positions


# ============================================================
# Extraction Method 1: Basic (embedding at target position)
# ============================================================

def extract_basic(model, tokenizer, sentence: str, target_word: str, layer: int = -1):
    """Extract embedding at target word position. Works for any model."""
    
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    token_positions = find_target_word_tokens(sentence, target_word, tokenizer)
    
    if not token_positions:
        return None
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    word_embs = hidden_states[0, token_positions, :]
    embedding = word_embs.mean(dim=0)
    
    return embedding


# ============================================================
# Extraction Method 2: Attention-Weighted Value Flow
# ============================================================

def extract_attention_flow(model, tokenizer, sentence: str, target_word: str, layer: int = -1):
    """
    Extract what information about the target word reached the end of sentence.
    
    For each attention head:
        contribution = attention[last_pos → target_pos] * V[target_pos]
    
    Then aggregate across heads.
    
    This captures "what the model concluded about this word given full context"
    """
    
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    token_positions = find_target_word_tokens(sentence, target_word, tokenizer)
    
    if not token_positions:
        return None
    
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1
    
    # Handle if target word is at the end
    if last_pos in token_positions:
        last_pos = min(token_positions) - 1
        if last_pos < 0:
            return None
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            output_attentions=True
        )
    
    # Get attention weights and hidden states at specified layer
    # attentions[layer] shape: (batch, n_heads, seq_len, seq_len)
    # hidden_states[layer] shape: (batch, seq_len, hidden_dim)
    
    if layer == -1:
        layer_idx = len(outputs.attentions) - 1
    else:
        layer_idx = layer
    
    attention = outputs.attentions[layer_idx]  # (1, n_heads, seq_len, seq_len)
    hidden = outputs.hidden_states[layer_idx]   # (1, seq_len, hidden_dim)
    
    # Attention from last position to target word positions
    # attention[0, :, last_pos, target_pos] = (n_heads,) for each target token
    
    # Get attention weights from last_pos to each target token, then average
    attn_to_target = attention[0, :, last_pos, token_positions].mean(dim=1)  # (n_heads,)
    
    # Get the value/hidden state at target position (average if multiple tokens)
    target_hidden = hidden[0, token_positions, :].mean(dim=0)  # (hidden_dim,)
    
    # Weighted contribution: how much of target's info reached the end
    # We weight by the mean attention across heads
    mean_attention = attn_to_target.mean()
    
    # Option A: Just return target hidden weighted by attention
    # This gives us (hidden_dim,) representing attended information
    contribution = mean_attention * target_hidden
    
    return contribution


def extract_attention_flow_v2(model, tokenizer, sentence: str, target_word: str):
    """
    Alternative: Collect attention patterns across ALL layers as features.
    
    Returns: For each layer and head, how much did final position attend to target?
    Shape: (n_layers * n_heads,)
    """
    
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    token_positions = find_target_word_tokens(sentence, target_word, tokenizer)
    
    if not token_positions:
        return None
    
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1
    
    if last_pos in token_positions:
        last_pos = min(token_positions) - 1
        if last_pos < 0:
            return None
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )
    
    # Collect attention to target from final position across all layers/heads
    attention_features = []
    
    for layer_attention in outputs.attentions:
        # layer_attention: (1, n_heads, seq_len, seq_len)
        attn_to_target = layer_attention[0, :, last_pos, token_positions].mean(dim=1)  # (n_heads,)
        attention_features.append(attn_to_target)
    
    # Concatenate all layers
    features = torch.cat(attention_features, dim=0)  # (n_layers * n_heads,)
    
    return features


def extract_attention_flow_v3(model, tokenizer, sentence: str, target_word: str, layer: int = -1):
    """
    V3: Reconstruct what information flowed from target to end position.
    
    In transformers, the contribution of position j to position i is:
        contribution_j→i = softmax(Q_i · K_j / sqrt(d)) * V_j
    
    We sum this across all heads to get the total information flow.
    """
    
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    token_positions = find_target_word_tokens(sentence, target_word, tokenizer)
    
    if not token_positions:
        return None
    
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1
    
    # Don't use last_pos if it's the target word itself
    if last_pos in token_positions:
        # Use position right before target word
        last_pos = min(token_positions) - 1
        if last_pos < 0:
            return None
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            output_attentions=True
        )
    
    if layer == -1:
        layer_idx = len(outputs.attentions) - 1
    else:
        layer_idx = layer
    
    # Attention: (1, n_heads, seq_len, seq_len)
    attention = outputs.attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)
    
    # We need the Value vectors. In GPT-2, we can approximate with hidden states.
    # The hidden state at layer L is roughly the output after attention at layer L.
    # The input to attention (which V is computed from) is hidden state at layer L-1.
    
    if layer_idx > 0:
        value_source = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
    else:
        value_source = outputs.hidden_states[0]
    
    # For each head, compute: attention[head, last_pos, target_pos] * V[target_pos]
    # Then sum/mean across heads
    
    n_heads = attention.shape[0]
    hidden_dim = value_source.shape[-1]
    head_dim = hidden_dim // n_heads
    
    # Get attention weights from last position to target positions
    # Shape: (n_heads, n_target_tokens)
    attn_weights = attention[:, last_pos, token_positions]  # (n_heads, n_target)
    
    # Average over target tokens if word is split
    attn_weights = attn_weights.mean(dim=1)  # (n_heads,)
    
    # Get value at target position (average if multiple tokens)
    target_value = value_source[0, token_positions, :].mean(dim=0)  # (hidden_dim,)
    
    # Weight by total attention
    total_attention = attn_weights.sum()
    contribution = total_attention * target_value
    
    return contribution


# ============================================================
# Extraction Method 3: End of sentence embedding
# ============================================================

def extract_end_of_sentence(model, tokenizer, sentence: str, target_word: str, layer: int = -1):
    """Just get the final token's embedding (contains full context)."""
    
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    hidden_states = outputs.hidden_states[layer]
    last_embedding = hidden_states[0, -1, :]
    
    return last_embedding


# ============================================================
# Batch extraction
# ============================================================

def extract_embeddings_batch(model, tokenizer, dataset, method="basic", layer=-1, device="cpu"):
    """Extract embeddings using specified method."""
    
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels = []
    skipped = 0
    
    for example in dataset.examples:
        try:
            if method == "basic":
                emb = extract_basic(model, tokenizer, example.text, dataset.word, layer)
            elif method == "attention_flow":
                emb = extract_attention_flow_v3(model, tokenizer, example.text, dataset.word, layer)
            elif method == "attention_pattern":
                emb = extract_attention_flow_v2(model, tokenizer, example.text, dataset.word)
            elif method == "end_of_sentence":
                emb = extract_end_of_sentence(model, tokenizer, example.text, dataset.word, layer)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if emb is None:
                skipped += 1
                continue
            
            embeddings.append(emb.cpu())
            labels.append(example.sense_id)
            
        except Exception as e:
            print(f"  Error: {e}")
            skipped += 1
            continue
    
    if not embeddings:
        return None, None
    
    return torch.stack(embeddings), torch.tensor(labels)


# ============================================================
# Evaluation (simple logistic regression)
# ============================================================

def evaluate(embeddings, labels, name=""):
    """Cross-validated logistic regression accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    X = embeddings.numpy()
    y = labels.numpy()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, len(np.unique(y)) * 2))
    
    mean_acc = scores.mean()
    std_acc = scores.std()
    
    return mean_acc, std_acc


# ============================================================
# Main comparison
# ============================================================

def main():
    from transformers import AutoModel, AutoTokenizer

    print("=" * 70)
    print("COMPARING EMBEDDING EXTRACTION METHODS")
    print("=" * 70)

    # Load datasets
    datasets = load_datasets("polysemous_datasets.json")

    results = {}

    # =========== GPT-2 ===========
    print("\n" + "=" * 70)
    print("MODEL: GPT-2")
    print("=" * 70)

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use attn_implementation='eager' to get attention weights
    # (Flash/SDPA attention doesn't return weights)
    model = AutoModel.from_pretrained(model_name, attn_implementation='eager')
    tokenizer.pad_token = tokenizer.eos_token
    
    n_layers = model.config.num_hidden_layers
    
    for word in ["bank", "bat", "crane", "cell", "spring", "match"]:
        print(f"\n--- {word.upper()} ({datasets[word].senses}) ---")
        
        results[f"gpt2_{word}"] = {}
        
        # Method 1: Basic (at target position)
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word], 
                                            method="basic", layer=-1)
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  Basic (target pos):     {acc:.3f} (+/- {std:.3f})")
            results[f"gpt2_{word}"]["basic"] = acc
        
        # Method 2: Attention flow
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word],
                                            method="attention_flow", layer=-1)
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  Attention flow:         {acc:.3f} (+/- {std:.3f})")
            results[f"gpt2_{word}"]["attention_flow"] = acc
        
        # Method 3: Attention patterns as features
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word],
                                            method="attention_pattern")
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  Attention patterns:     {acc:.3f} (+/- {std:.3f})")
            results[f"gpt2_{word}"]["attention_pattern"] = acc
        
        # Method 4: End of sentence
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word],
                                            method="end_of_sentence", layer=-1)
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  End of sentence:        {acc:.3f} (+/- {std:.3f})")
            results[f"gpt2_{word}"]["end_of_sentence"] = acc
    
    # =========== BERT ===========
    print("\n" + "=" * 70)
    print("MODEL: BERT")
    print("=" * 70)
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    for word in ["bank", "bat", "crane", "cell", "spring", "match"]:
        print(f"\n--- {word.upper()} ({datasets[word].senses}) ---")
        
        results[f"bert_{word}"] = {}
        
        # BERT: Basic at target position (has full context!)
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word],
                                            method="basic", layer=-1)
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  Basic (target pos):     {acc:.3f} (+/- {std:.3f})")
            results[f"bert_{word}"]["basic"] = acc
        
        # BERT: Also try [CLS] token (index 0)
        emb, lab = extract_embeddings_batch(model, tokenizer, datasets[word],
                                            method="end_of_sentence", layer=-1)
        if emb is not None:
            acc, std = evaluate(emb, lab)
            print(f"  [CLS] token:            {acc:.3f} (+/- {std:.3f})")
            results[f"bert_{word}"]["cls"] = acc
    
    # =========== Summary ===========
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    print(f"\n{'Word':<10} | {'GPT2-Basic':<12} | {'GPT2-AttnFlow':<14} | {'GPT2-AttnPat':<13} | {'GPT2-EOS':<10} | {'BERT':<10}")
    print("-" * 85)
    
    for word in ["bank", "bat", "crane", "cell", "spring", "match"]:
        gpt2_basic = results.get(f"gpt2_{word}", {}).get("basic", 0)
        gpt2_flow = results.get(f"gpt2_{word}", {}).get("attention_flow", 0)
        gpt2_pat = results.get(f"gpt2_{word}", {}).get("attention_pattern", 0)
        gpt2_eos = results.get(f"gpt2_{word}", {}).get("end_of_sentence", 0)
        bert = results.get(f"bert_{word}", {}).get("basic", 0)
        
        print(f"{word:<10} | {gpt2_basic:<12.3f} | {gpt2_flow:<14.3f} | {gpt2_pat:<13.3f} | {gpt2_eos:<10.3f} | {bert:<10.3f}")


if __name__ == "__main__":
    main()

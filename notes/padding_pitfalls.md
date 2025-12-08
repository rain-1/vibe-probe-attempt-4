# Batched Hidden State Extraction: Padding Pitfalls

## The Problem

When extracting hidden states from a causal language model in batches, **padding corrupts the hidden states** in subtle ways that are easy to miss but devastating for probe training.

### Why This Happens

1. **Padding tokens affect attention**: Even with an attention mask, the model's internal computations can be affected by padding tokens in ways that change the hidden states.

2. **Causal attention + padding**: In left-to-right causal models, padding is typically added on the RIGHT. But the attention patterns for tokens before padding may still differ from single-sample inference.

3. **Layer normalization**: RMSNorm/LayerNorm statistics can be subtly affected by the presence of padding tokens in the sequence.

### Symptoms

- Hidden states extracted in batches have lower cosine similarity to single-sample extraction
- Using `lm_head` directly on batched-extracted hidden states gives wrong predictions
- Probes trained on batched data learn to compensate for corruption, not the true pattern
- High validation accuracy but poor generalization

### Verification Method

Extract same sample both ways and compare:

```python
# Single sample (ground truth)
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
single_hidden = outputs.hidden_states[-1][0, -1]

# Batched (potentially corrupted)  
inputs = tokenizer([text, other_text], return_tensors="pt", padding=True)
outputs = model(**inputs, output_hidden_states=True)
# Find last non-padded position
seq_len = inputs.attention_mask[0].sum().item()
batched_hidden = outputs.hidden_states[-1][0, seq_len - 1]

# Compare
cos_sim = F.cosine_similarity(single_hidden, batched_hidden, dim=0)
# Should be > 0.9999 if extraction is correct
```

### Ground Truth Test

Use `lm_head` to verify hidden states are correct:

```python
# If hidden state is correct, this should match model's actual prediction
logits = model.lm_head(hidden_state.unsqueeze(0))
probs = torch.softmax(logits, dim=-1)
# probs[0, target_token_id] should match what model predicts
```

## Solutions

### Solution 1: Single-sample extraction (slow but correct)

```python
for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1]  # Last token
```

### Solution 2: Left-padding (for causal LMs)

Causal LMs naturally handle left-padding better because the "real" tokens are at the end:

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # If no pad token

inputs = tokenizer(texts, return_tensors="pt", padding=True)
outputs = model(**inputs, output_hidden_states=True)

# With left-padding, last position is always the real last token
hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_dim)
```

### Solution 3: Batch by length (no padding needed)

Group texts by token length, batch same-length texts together:

```python
from itertools import groupby

# Tokenize all texts first
tokenized = [(text, tokenizer.encode(text)) for text in texts]

# Group by length
tokenized.sort(key=lambda x: len(x[1]))
for length, group in groupby(tokenized, key=lambda x: len(x[1])):
    batch_texts = [t[0] for t in group]
    # No padding needed - all same length
    inputs = tokenizer(batch_texts, return_tensors="pt")
    # ... extract hidden states
```

## Recommendation

For probe training where correctness matters more than speed:

1. **Always verify** a sample of batched extractions against single-sample ground truth
2. **Use left-padding** for causal language models
3. **Test with lm_head**: If `model.lm_head(hidden)` gives wrong predictions, your hidden states are wrong
4. When in doubt, use single-sample extraction

## Impact on This Project

Our 10k sample dataset was extracted with right-padding in batches. Testing showed:
- `lm_head` directly on stored hidden states: ~77% accuracy
- Trained linear probe on same data: ~86% accuracy

The probe "learned" to work around corrupted hidden states, but this means it learned spurious patterns that don't generalize to real inference (where we extract one sample at a time).

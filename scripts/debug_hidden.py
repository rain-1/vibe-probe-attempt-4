"""Debug: Compare stored hidden states to fresh extraction."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np

model = AutoModelForCausalLM.from_pretrained('unsloth/gemma-3-270m', torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('unsloth/gemma-3-270m')
model.eval()

df = pd.read_parquet('data/next_token_real_10k.parquet')

# Check a few positive samples
print("Checking positive samples (label=1, next token should be ' the'):")
for i, row in df[df.label == 1].head(5).iterrows():
    text = row['text']
    stored = np.array(row['layer-final-hidden'], dtype=np.float32)
    
    # Fresh extraction
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0, -1]  # Last position
        normed = model.model.norm(last_hidden.unsqueeze(0)).squeeze()
    
    fresh = normed.float().numpy()
    
    # Cosine similarity
    cos = np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh))
    
    # What does model predict with stored hidden?
    with torch.no_grad():
        logits = model.lm_head(torch.tensor(stored).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        p_the_stored = probs[0, 506].item()
        
        logits_fresh = model.lm_head(normed.unsqueeze(0))
        probs_fresh = torch.softmax(logits_fresh, dim=-1)
        p_the_fresh = probs_fresh[0, 506].item()
    
    print(f"  Text ends: ...{text[-30:]!r}")
    print(f"  Cos sim: {cos:.4f}, P(the) stored: {p_the_stored:.4f}, P(the) fresh: {p_the_fresh:.4f}")
    print()

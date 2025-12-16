"""
Contrastive Probe Training for Word Sense Disambiguation

Complete implementation - just run it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


# ============================================================
# Data Loading
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
# Embedding Extraction
# ============================================================

def find_target_word_tokens(sentence: str, target_word: str, tokenizer, inputs) -> List[int]:
    """Find token indices corresponding to the target word."""
    word_lower = target_word.lower()
    sentence_lower = sentence.lower()
    
    # Find character position of target word
    start_char = sentence_lower.find(word_lower)
    if start_char == -1:
        # Try plural
        start_char = sentence_lower.find(word_lower + 's')
        if start_char != -1:
            end_char = start_char + len(word_lower) + 1
        else:
            # Try with common suffixes
            for suffix in ['es', 'ed', 'ing']:
                start_char = sentence_lower.find(word_lower + suffix)
                if start_char != -1:
                    end_char = start_char + len(word_lower) + len(suffix)
                    break
            else:
                return []
    else:
        end_char = start_char + len(word_lower)
    
    # Get offset mapping if available
    if "offset_mapping" in inputs:
        offsets = inputs["offset_mapping"][0].tolist()
    else:
        # Re-tokenize with offset mapping
        temp_inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
        offsets = temp_inputs["offset_mapping"][0].tolist()
    
    # Find tokens that overlap with target word
    token_positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end == 0:  # Skip special tokens
            continue
        # Check for overlap
        if tok_start < end_char and tok_end > start_char:
            token_positions.append(idx)
    
    return token_positions


def extract_embeddings(model, tokenizer, dataset: PolysemousDataset, layer: int = -1, device: str = "cpu"):
    """Extract embeddings at target word position for all examples."""
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels = []
    skipped = 0
    
    for example in dataset.examples:
        try:
            # Tokenize
            inputs = tokenizer(
                example.text, 
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True
            )
            
            # Find target word tokens
            token_positions = find_target_word_tokens(
                example.text, dataset.word, tokenizer, inputs
            )
            
            if not token_positions:
                print(f"  Skipping (can't find '{dataset.word}'): {example.text[:50]}...")
                skipped += 1
                continue
            
            # Forward pass
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
            
            # Extract and average embeddings at target positions
            word_embs = hidden_states[0, token_positions, :]  # (n_tokens, hidden_dim)
            embedding = word_embs.mean(dim=0).cpu()  # (hidden_dim,)
            
            embeddings.append(embedding)
            labels.append(example.sense_id)
            
        except Exception as e:
            print(f"  Error on '{example.text[:50]}...': {e}")
            skipped += 1
            continue
    
    print(f"Extracted {len(embeddings)} embeddings, skipped {skipped}")
    
    if not embeddings:
        return None, None
    
    return torch.stack(embeddings), torch.tensor(labels)


# ============================================================
# Contrastive Learning
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss - pulls same-class together, pushes different apart."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)
            labels: (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # L2 normalize
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Positive mask (same label, not self)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        
        # Check if any positives exist
        pos_count = pos_mask.sum(dim=1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=device)
        
        # Numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Denominator: sum over all except self
        self_mask = torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log prob over positives
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        
        # Only count samples with positives
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        loss = -mean_log_prob[valid].mean()
        return loss


class ContrastiveProbe(nn.Module):
    """Linear projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def train_contrastive(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dim: int = 64,
    epochs: int = 200,
    lr: float = 0.001,
    temperature: float = 0.1,
    batch_size: int = None,  # None = full batch
) -> ContrastiveProbe:
    """Train contrastive probe."""
    
    input_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    
    probe = ContrastiveProbe(input_dim, output_dim)
    criterion = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    if batch_size is None:
        batch_size = n_samples  # Full batch
    
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        emb_shuffled = embeddings[perm]
        lab_shuffled = labels[perm]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_emb = emb_shuffled[i:i+batch_size]
            batch_lab = lab_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            projected = probe(batch_emb)
            loss = criterion(projected, batch_lab)
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
        
        if epoch % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}: loss = {avg_loss:.4f}")
    
    return probe


# ============================================================
# Evaluation
# ============================================================

def evaluate_linear(embeddings: torch.Tensor, labels: torch.Tensor, name: str = ""):
    """Evaluate with simple logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    X = embeddings.numpy()
    y = labels.numpy()
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, len(np.unique(y)) * 2))
    
    print(f"  {name} Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    return scores.mean()


def evaluate_contrastive(probe: ContrastiveProbe, embeddings: torch.Tensor, labels: torch.Tensor):
    """Evaluate contrastive probe by fitting classifier on projected space."""
    probe.eval()
    with torch.no_grad():
        projected = probe(embeddings)
    
    return evaluate_linear(projected, labels, name="Contrastive")


def visualize_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, senses: List[str], title: str, filename: str):
    """PCA visualization of embeddings."""
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(embeddings.numpy())
        
        plt.figure(figsize=(8, 6))
        for sense_id, sense_name in enumerate(senses):
            mask = labels.numpy() == sense_id
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=sense_name, alpha=0.7, s=50)
        
        plt.legend()
        plt.title(title)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")
    except ImportError:
        print("  (matplotlib not available for visualization)")


# ============================================================
# Main
# ============================================================

def main():
    from transformers import AutoModel, AutoTokenizer
    
    print("Loading model...")
    model_name = "gpt2"  # Change this to try different models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    n_layers = model.config.num_hidden_layers
    print(f"Model: {model_name} ({n_layers} layers, {model.config.hidden_size} hidden dim)")
    
    print("\nLoading datasets...")
    datasets = load_datasets("polysemous_datasets.json")
    
    # Test on each word
    results = {}
    
    for word in ["bank", "bat", "crane", "cell", "spring", "match"]:
        print(f"\n{'='*60}")
        print(f"WORD: {word.upper()}")
        print(f"{'='*60}")
        
        dataset = datasets[word]
        print(dataset.summary())
        
        word_results = {}
        
        # Try different layers
        layers_to_try = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, -1]
        
        for layer_idx in layers_to_try:
            layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "last_layer"
            print(f"\n--- {layer_name} ---")
            
            # Extract embeddings
            embeddings, labels = extract_embeddings(model, tokenizer, dataset, layer=layer_idx)
            
            if embeddings is None:
                print("  No embeddings extracted!")
                continue
            
            # Visualize raw embeddings
            visualize_embeddings(
                embeddings, labels, dataset.senses,
                f"{word} - {layer_name} (raw)",
                f"{word}_{layer_name}_raw.png"
            )
            
            # Baseline: logistic regression on raw embeddings
            print("\n  Baseline (raw embeddings):")
            baseline_acc = evaluate_linear(embeddings, labels, name="Logistic Reg")
            
            # Contrastive probe
            print("\n  Training contrastive probe...")
            probe = train_contrastive(
                embeddings, labels,
                output_dim=32,
                epochs=200,
                lr=0.001,
                temperature=0.1
            )
            
            print("\n  After contrastive training:")
            contrastive_acc = evaluate_contrastive(probe, embeddings, labels)
            
            # Visualize projected embeddings
            probe.eval()
            with torch.no_grad():
                projected = probe(embeddings)
            visualize_embeddings(
                projected, labels, dataset.senses,
                f"{word} - {layer_name} (after contrastive)",
                f"{word}_{layer_name}_contrastive.png"
            )
            
            word_results[layer_name] = {
                "baseline": baseline_acc,
                "contrastive": contrastive_acc
            }
        
        results[word] = word_results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for word, word_results in results.items():
        print(f"\n{word}:")
        for layer, accs in word_results.items():
            print(f"  {layer}: baseline={accs['baseline']:.3f}, contrastive={accs['contrastive']:.3f}")


if __name__ == "__main__":
    main()

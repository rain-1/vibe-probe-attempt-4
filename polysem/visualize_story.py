"""
Visualize Contrastive Probes on Story Text

Applies trained contrastive probes to identify word senses in a story,
and creates visualizations showing the predicted senses.
"""

import torch
import torch.nn.functional as F
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from contrastive_probe import (
    load_datasets,
    extract_embeddings,
    train_contrastive,
    ContrastiveProbe,
    PolysemousDataset
)


@dataclass
class WordOccurrence:
    """A single occurrence of a polysemous word in the story."""
    word: str
    start_char: int
    end_char: int
    sentence: str
    paragraph_num: int
    predicted_sense: str = None
    confidence: float = 0.0
    embedding: torch.Tensor = None


def find_word_occurrences(story: str, target_words: List[str]) -> List[WordOccurrence]:
    """Find all occurrences of target words in the story."""
    occurrences = []
    paragraphs = story.strip().split('\n')

    for para_idx, paragraph in enumerate(paragraphs):
        for word in target_words:
            # Find all occurrences (including plurals and verb forms)
            patterns = [
                rf'\b{word}\b',
                rf'\b{word}s\b',
                rf'\b{word}es\b',
                rf'\b{word}ed\b',
                rf'\b{word}ing\b',
            ]

            seen_positions = set()  # Avoid duplicates from overlapping patterns

            for pattern in patterns:
                for match in re.finditer(pattern, paragraph, re.IGNORECASE):
                    if match.start() not in seen_positions:
                        seen_positions.add(match.start())
                        occurrences.append(WordOccurrence(
                            word=word,
                            start_char=match.start(),
                            end_char=match.end(),
                            sentence=paragraph,
                            paragraph_num=para_idx + 1
                        ))

    return occurrences


def find_token_positions_for_char_span(
    sentence: str,
    start_char: int,
    end_char: int,
    tokenizer,
    inputs
) -> List[int]:
    """Find token indices for a specific character span."""
    # Get offset mapping
    if "offset_mapping" in inputs:
        offsets = inputs["offset_mapping"][0].tolist()
    else:
        temp_inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
        offsets = temp_inputs["offset_mapping"][0].tolist()

    # Find tokens that overlap with the character span
    token_positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end == 0:  # Skip special tokens
            continue
        # Check for overlap
        if tok_start < end_char and tok_end > start_char:
            token_positions.append(idx)

    return token_positions


def extract_story_embeddings(
    model,
    tokenizer,
    occurrences: List[WordOccurrence],
    layer: int = -1,
    device: str = "cpu"
) -> List[WordOccurrence]:
    """Extract embeddings for each word occurrence in the story."""

    model = model.to(device)
    model.eval()

    # Cache sentence embeddings to avoid recomputing
    sentence_cache = {}

    for occ in occurrences:
        try:
            # Get or compute hidden states for this sentence
            if occ.sentence not in sentence_cache:
                inputs = tokenizer(
                    occ.sentence,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    padding=True,
                    truncation=True
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                sentence_cache[occ.sentence] = {
                    'hidden_states': outputs.hidden_states[layer].cpu(),
                    'inputs': inputs
                }

            cached = sentence_cache[occ.sentence]

            # Find tokens for THIS specific occurrence
            token_positions = find_token_positions_for_char_span(
                occ.sentence, occ.start_char, occ.end_char,
                tokenizer, cached['inputs']
            )

            if not token_positions:
                print(f"  Warning: No tokens found for '{occ.word}' at pos {occ.start_char}")
                continue

            hidden_states = cached['hidden_states']
            word_embs = hidden_states[0, token_positions, :]
            occ.embedding = word_embs.mean(dim=0)

        except Exception as e:
            print(f"Error extracting embedding for '{occ.word}': {e}")
            continue

    return [o for o in occurrences if o.embedding is not None]


def predict_senses(
    occurrences: List[WordOccurrence],
    probes: Dict[str, ContrastiveProbe],
    train_embeddings: Dict[str, torch.Tensor],
    train_labels: Dict[str, torch.Tensor],
    datasets: Dict[str, PolysemousDataset]
) -> List[WordOccurrence]:
    """Predict sense for each occurrence using trained probes + kNN."""

    for occ in occurrences:
        if occ.word not in probes:
            continue

        probe = probes[occ.word]
        train_emb = train_embeddings[occ.word]
        train_lab = train_labels[occ.word]
        dataset = datasets[occ.word]

        probe.eval()
        with torch.no_grad():
            # Project story embedding
            proj_story = probe(occ.embedding.unsqueeze(0))
            proj_story = F.normalize(proj_story, dim=1)

            # Project training embeddings
            proj_train = probe(train_emb)
            proj_train = F.normalize(proj_train, dim=1)

        # Use kNN for classification
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(proj_train.numpy(), train_lab.numpy())

        pred = knn.predict(proj_story.numpy())[0]
        proba = knn.predict_proba(proj_story.numpy())[0]

        occ.predicted_sense = dataset.senses[pred]
        occ.confidence = proba[pred]

    return occurrences


def create_2d_projection(
    occurrences: List[WordOccurrence],
    probes: Dict[str, ContrastiveProbe],
    train_embeddings: Dict[str, torch.Tensor],
    train_labels: Dict[str, torch.Tensor],
    datasets: Dict[str, PolysemousDataset]
) -> Dict[str, dict]:
    """Create 2D PCA projections for visualization."""

    projections = {}

    for word in probes.keys():
        probe = probes[word]
        train_emb = train_embeddings[word]
        train_lab = train_labels[word]
        dataset = datasets[word]

        # Get story occurrences for this word
        word_occs = [o for o in occurrences if o.word == word and o.embedding is not None]

        if not word_occs:
            continue

        probe.eval()
        with torch.no_grad():
            # Project all embeddings
            proj_train = probe(train_emb)
            story_embs = torch.stack([o.embedding for o in word_occs])
            proj_story = probe(story_embs)

        # Combine for PCA
        all_proj = torch.cat([proj_train, proj_story], dim=0)

        # PCA to 2D
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_proj.numpy())

        n_train = len(train_emb)
        train_2d = all_2d[:n_train]
        story_2d = all_2d[n_train:]

        projections[word] = {
            'train_2d': train_2d,
            'train_labels': train_lab.numpy(),
            'story_2d': story_2d,
            'story_occs': word_occs,
            'senses': dataset.senses,
            'pca_variance': pca.explained_variance_ratio_
        }

    return projections


def plot_story_visualization(
    projections: Dict[str, dict],
    output_file: str = "story_sense_visualization.png"
):
    """Create a comprehensive visualization of word senses in the story."""

    words = list(projections.keys())
    n_words = len(words)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Color schemes for each word's senses
    sense_colors = {
        'bank': {'financial_institution': '#2ecc71', 'river_edge': '#3498db'},
        'bat': {'animal': '#9b59b6', 'sports_equipment': '#e74c3c'},
        'crane': {'bird': '#f39c12', 'machine': '#1abc9c'},
        'cell': {'biology': '#e91e63', 'prison': '#607d8b', 'phone': '#00bcd4'},
        'spring': {'season': '#8bc34a', 'coil': '#ff5722', 'water_source': '#03a9f4'},
        'match': {'fire_starter': '#ff9800', 'competition': '#673ab7', 'to_pair': '#009688'},
    }

    for idx, word in enumerate(words):
        ax = axes[idx]
        data = projections[word]

        train_2d = data['train_2d']
        train_labels = data['train_labels']
        story_2d = data['story_2d']
        story_occs = data['story_occs']
        senses = data['senses']
        pca_var = data['pca_variance']

        colors = sense_colors.get(word, {})

        # Plot training data (small, semi-transparent)
        for sense_idx, sense_name in enumerate(senses):
            mask = train_labels == sense_idx
            color = colors.get(sense_name, f'C{sense_idx}')
            ax.scatter(
                train_2d[mask, 0],
                train_2d[mask, 1],
                c=color,
                alpha=0.3,
                s=30,
                label=f'{sense_name} (train)'
            )

        # Plot story occurrences (large, with paragraph labels)
        for i, occ in enumerate(story_occs):
            color = colors.get(occ.predicted_sense, 'gray')
            ax.scatter(
                story_2d[i, 0],
                story_2d[i, 1],
                c=color,
                s=200,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                zorder=10
            )
            # Add paragraph number label
            ax.annotate(
                f'P{occ.paragraph_num}',
                (story_2d[i, 0], story_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold'
            )

        ax.set_title(f'{word.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_var[1]:.1%})')

        # Create legend
        handles = []
        for sense_name in senses:
            color = colors.get(sense_name, 'gray')
            handles.append(mpatches.Patch(color=color, label=sense_name))
        handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                                   markersize=15, markeredgecolor='black', label='Story occurrence'))
        ax.legend(handles=handles, loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Contrastive Probe Word Sense Disambiguation\n(Story occurrences marked with ★)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_file}")


def create_story_annotation(
    story: str,
    occurrences: List[WordOccurrence],
    output_file: str = "story_annotated.html"
):
    """Create an HTML file with color-coded word senses."""

    sense_colors = {
        'financial_institution': '#2ecc71',
        'river_edge': '#3498db',
        'animal': '#9b59b6',
        'sports_equipment': '#e74c3c',
        'bird': '#f39c12',
        'machine': '#1abc9c',
        'biology': '#e91e63',
        'prison': '#607d8b',
        'phone': '#00bcd4',
        'season': '#8bc34a',
        'coil': '#ff5722',
        'water_source': '#03a9f4',
        'fire_starter': '#ff9800',
        'competition': '#673ab7',
        'to_pair': '#009688',
    }

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Word Sense Disambiguation - Story Visualization</title>
    <style>
        body { font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.8; }
        h1 { text-align: center; }
        .paragraph { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }
        .word { padding: 2px 6px; border-radius: 4px; font-weight: bold; cursor: help; }
        .legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; padding: 15px; background: #eee; border-radius: 8px; }
        .legend-item { display: flex; align-items: center; gap: 5px; font-size: 14px; }
        .legend-color { width: 20px; height: 20px; border-radius: 4px; }
        .legend-section { margin-bottom: 10px; }
        .legend-title { font-weight: bold; width: 100%; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Word Sense Disambiguation</h1>
    <h3 style="text-align: center; color: #666;">Contrastive Probe Predictions</h3>

    <div class="legend">
"""

    # Group senses by word
    word_senses = {
        'bank': ['financial_institution', 'river_edge'],
        'bat': ['animal', 'sports_equipment'],
        'crane': ['bird', 'machine'],
        'cell': ['biology', 'prison', 'phone'],
        'spring': ['season', 'coil', 'water_source'],
        'match': ['fire_starter', 'competition', 'to_pair'],
    }

    for word, senses in word_senses.items():
        html += f'<div class="legend-section"><span class="legend-title">{word.upper()}:</span>'
        for sense in senses:
            color = sense_colors[sense]
            html += f'<div class="legend-item"><div class="legend-color" style="background: {color}"></div>{sense.replace("_", " ")}</div>'
        html += '</div>'

    html += "</div>\n"

    # Process paragraphs
    paragraphs = story.strip().split('\n')

    for para_idx, paragraph in enumerate(paragraphs):
        # Get occurrences for this paragraph
        para_occs = [o for o in occurrences if o.paragraph_num == para_idx + 1]

        # Sort by position (reverse to replace from end to start)
        para_occs_sorted = sorted(para_occs, key=lambda x: x.start_char, reverse=True)

        annotated = paragraph
        for occ in para_occs_sorted:
            if occ.predicted_sense:
                color = sense_colors.get(occ.predicted_sense, '#999')
                word_text = annotated[occ.start_char:occ.end_char]
                replacement = f'<span class="word" style="background: {color}; color: white;" title="{occ.predicted_sense} ({occ.confidence:.0%})">{word_text}</span>'
                annotated = annotated[:occ.start_char] + replacement + annotated[occ.end_char:]

        html += f'<div class="paragraph"><strong>Paragraph {para_idx + 1}:</strong> {annotated}</div>\n'

    html += """
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Saved annotated story: {output_file}")


def print_predictions_summary(occurrences: List[WordOccurrence]):
    """Print a summary of all predictions."""
    print("\n" + "="*70)
    print("PREDICTIONS SUMMARY")
    print("="*70)

    # Group by word
    words = sorted(set(o.word for o in occurrences))

    for word in words:
        word_occs = sorted([o for o in occurrences if o.word == word],
                          key=lambda x: (x.paragraph_num, x.start_char))
        print(f"\n{word.upper()} ({len(word_occs)} occurrences):")
        for occ in word_occs:
            # Get context around the word
            ctx_start = max(0, occ.start_char - 30)
            ctx_end = min(len(occ.sentence), occ.end_char + 30)
            context = occ.sentence[ctx_start:ctx_end]
            if ctx_start > 0:
                context = "..." + context
            if ctx_end < len(occ.sentence):
                context = context + "..."

            matched_word = occ.sentence[occ.start_char:occ.end_char]
            print(f"  P{occ.paragraph_num}: '{matched_word}' -> {occ.predicted_sense} ({occ.confidence:.0%})")
            print(f"      \"{context}\"")


def main():
    from transformers import AutoModel, AutoTokenizer

    print("="*70)
    print("STORY WORD SENSE VISUALIZATION")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets and story
    print("Loading datasets...")
    datasets = load_datasets("polysemous_datasets.json")

    print("Loading story...")
    with open("claudes polysem story.txt", "r") as f:
        story = f.read()

    target_words = ["bank", "bat", "crane", "cell", "spring", "match"]

    # Train probes and store embeddings
    print("\nTraining contrastive probes...")
    probes = {}
    train_embeddings = {}
    train_labels = {}

    for word in target_words:
        print(f"\n  Training probe for '{word}'...")
        dataset = datasets[word]

        embeddings, labels = extract_embeddings(model, tokenizer, dataset, layer=-1)

        if embeddings is None:
            print(f"    Skipping {word} - no embeddings")
            continue

        probe = train_contrastive(
            embeddings, labels,
            output_dim=32,
            epochs=200,
            lr=0.001,
            temperature=0.1
        )

        probes[word] = probe
        train_embeddings[word] = embeddings
        train_labels[word] = labels

    # Find word occurrences in story
    print("\nFinding word occurrences in story...")
    occurrences = find_word_occurrences(story, target_words)
    print(f"  Found {len(occurrences)} total occurrences")

    # Extract embeddings for story occurrences
    print("\nExtracting story embeddings...")
    occurrences = extract_story_embeddings(model, tokenizer, occurrences, layer=-1)
    print(f"  Extracted {len(occurrences)} embeddings")

    # Predict senses
    print("\nPredicting senses...")
    occurrences = predict_senses(occurrences, probes, train_embeddings, train_labels, datasets)

    # Print summary
    print_predictions_summary(occurrences)

    # Create 2D projections
    print("\nCreating visualizations...")
    projections = create_2d_projection(occurrences, probes, train_embeddings, train_labels, datasets)

    # Plot
    plot_story_visualization(projections, "story_sense_visualization.png")

    # Create annotated HTML
    create_story_annotation(story, occurrences, "story_annotated.html")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nOutputs:")
    print("  - story_sense_visualization.png  (PCA scatter plots)")
    print("  - story_annotated.html           (color-coded story)")


if __name__ == "__main__":
    main()

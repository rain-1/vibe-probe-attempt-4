"""
Visualize all extraction methods on the story for comparison.

Generates separate HTML files for each method:
- GPT-2 Basic (target position)
- GPT-2 Attention Flow (what flowed from target to end)
- GPT-2 End-of-Sentence
- RoBERTa (bidirectional)
"""

import torch
import torch.nn.functional as F
import json
import re
from dataclasses import dataclass
from typing import List, Dict
from sklearn.neighbors import KNeighborsClassifier

from contrastive_probe import (
    load_datasets,
    train_contrastive,
    ContrastiveProbe,
    PolysemousDataset
)


@dataclass
class WordOccurrence:
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
            patterns = [
                rf'\b{word}\b',
                rf'\b{word}s\b',
                rf'\b{word}es\b',
                rf'\b{word}ed\b',
                rf'\b{word}ing\b',
            ]

            seen_positions = set()

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


def find_token_positions_for_char_span(sentence, start_char, end_char, tokenizer, inputs):
    """Find token indices for a specific character span."""
    if "offset_mapping" in inputs:
        offsets = inputs["offset_mapping"][0].tolist()
    else:
        temp_inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
        offsets = temp_inputs["offset_mapping"][0].tolist()

    token_positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end == 0:
            continue
        if tok_start < end_char and tok_end > start_char:
            token_positions.append(idx)

    return token_positions


# ============================================================
# Extraction Methods
# ============================================================

def extract_basic(model, tokenizer, occurrences, layer=-1, device="cpu"):
    """Extract embedding at target word position."""
    model = model.to(device)
    model.eval()

    sentence_cache = {}

    for occ in occurrences:
        try:
            if occ.sentence not in sentence_cache:
                inputs = tokenizer(
                    occ.sentence,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    padding=True,
                    truncation=True
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True
                    )

                sentence_cache[occ.sentence] = {
                    'hidden_states': outputs.hidden_states[layer].cpu(),
                    'inputs': inputs
                }

            cached = sentence_cache[occ.sentence]
            token_positions = find_token_positions_for_char_span(
                occ.sentence, occ.start_char, occ.end_char,
                tokenizer, cached['inputs']
            )

            if not token_positions:
                continue

            word_embs = cached['hidden_states'][0, token_positions, :]
            occ.embedding = word_embs.mean(dim=0)

        except Exception as e:
            print(f"Error: {e}")
            continue

    return [o for o in occurrences if o.embedding is not None]


def extract_attention_flow(model, tokenizer, occurrences, layer=-1, device="cpu"):
    """Extract what information about target word reached end of sentence."""
    model = model.to(device)
    model.eval()

    sentence_cache = {}

    for occ in occurrences:
        try:
            if occ.sentence not in sentence_cache:
                inputs = tokenizer(
                    occ.sentence,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    padding=True,
                    truncation=True
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True,
                        output_attentions=True
                    )

                sentence_cache[occ.sentence] = {
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions,
                    'inputs': inputs,
                    'seq_len': inputs["input_ids"].shape[1]
                }

            cached = sentence_cache[occ.sentence]
            token_positions = find_token_positions_for_char_span(
                occ.sentence, occ.start_char, occ.end_char,
                tokenizer, cached['inputs']
            )

            if not token_positions:
                continue

            seq_len = cached['seq_len']
            last_pos = seq_len - 1

            # Don't use last_pos if it's the target word
            if last_pos in token_positions:
                last_pos = min(token_positions) - 1
                if last_pos < 0:
                    continue

            layer_idx = layer if layer >= 0 else len(cached['attentions']) + layer

            attention = cached['attentions'][layer_idx][0].cpu()  # (n_heads, seq, seq)
            hidden = cached['hidden_states'][layer_idx][0].cpu()  # (seq, hidden)

            # Attention from last position to target positions
            attn_weights = attention[:, last_pos, token_positions].mean(dim=1)  # (n_heads,)

            # Get value at target position
            target_value = hidden[token_positions, :].mean(dim=0)  # (hidden_dim,)

            # Weight by total attention
            total_attention = attn_weights.sum()
            occ.embedding = total_attention * target_value

        except Exception as e:
            print(f"Error: {e}")
            continue

    return [o for o in occurrences if o.embedding is not None]


def extract_end_of_sentence(model, tokenizer, occurrences, layer=-1, device="cpu"):
    """Extract the final token's embedding."""
    model = model.to(device)
    model.eval()

    sentence_cache = {}

    for occ in occurrences:
        try:
            if occ.sentence not in sentence_cache:
                inputs = tokenizer(
                    occ.sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True
                    )

                sentence_cache[occ.sentence] = {
                    'last_hidden': outputs.hidden_states[layer][0, -1, :].cpu()
                }

            occ.embedding = sentence_cache[occ.sentence]['last_hidden']

        except Exception as e:
            print(f"Error: {e}")
            continue

    return [o for o in occurrences if o.embedding is not None]


# ============================================================
# Training and Prediction
# ============================================================

def extract_training_embeddings(model, tokenizer, dataset, method, layer=-1, device="cpu"):
    """Extract embeddings for training data using specified method."""
    from contrastive_probe import find_target_word_tokens

    model = model.to(device)
    model.eval()

    embeddings = []
    labels = []

    for example in dataset.examples:
        try:
            inputs = tokenizer(
                example.text,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True
            )

            token_positions = find_target_word_tokens(example.text, dataset.word, tokenizer, inputs)

            if not token_positions:
                continue

            if method == "basic":
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True
                    )
                hidden = outputs.hidden_states[layer]
                emb = hidden[0, token_positions, :].mean(dim=0).cpu()

            elif method == "attention_flow":
                seq_len = inputs["input_ids"].shape[1]
                last_pos = seq_len - 1

                if last_pos in token_positions:
                    last_pos = min(token_positions) - 1
                    if last_pos < 0:
                        continue

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True,
                        output_attentions=True
                    )

                layer_idx = layer if layer >= 0 else len(outputs.attentions) + layer
                attention = outputs.attentions[layer_idx][0].cpu()
                hidden = outputs.hidden_states[layer_idx][0].cpu()

                attn_weights = attention[:, last_pos, token_positions].mean(dim=1)
                target_value = hidden[token_positions, :].mean(dim=0)
                total_attention = attn_weights.sum()
                emb = total_attention * target_value

            elif method == "end_of_sentence":
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True
                    )
                emb = outputs.hidden_states[layer][0, -1, :].cpu()

            embeddings.append(emb)
            labels.append(example.sense_id)

        except Exception as e:
            continue

    if not embeddings:
        return None, None

    return torch.stack(embeddings), torch.tensor(labels)


def predict_senses(occurrences, probes, train_embeddings, train_labels, datasets):
    """Predict sense for each occurrence."""
    for occ in occurrences:
        if occ.word not in probes:
            continue

        probe = probes[occ.word]
        train_emb = train_embeddings[occ.word]
        train_lab = train_labels[occ.word]
        dataset = datasets[occ.word]

        probe.eval()
        with torch.no_grad():
            proj_story = probe(occ.embedding.unsqueeze(0))
            proj_story = F.normalize(proj_story, dim=1)
            proj_train = probe(train_emb)
            proj_train = F.normalize(proj_train, dim=1)

        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(proj_train.numpy(), train_lab.numpy())

        pred = knn.predict(proj_story.numpy())[0]
        proba = knn.predict_proba(proj_story.numpy())[0]

        occ.predicted_sense = dataset.senses[pred]
        occ.confidence = proba[pred]

    return occurrences


# ============================================================
# HTML Generation
# ============================================================

def create_html(story, occurrences, method_name, output_file):
    """Create HTML visualization."""
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

    # Count correct/total (based on ground truth we know from the story)
    ground_truth = {
        # P1
        (1, 'spring', 5): 'season',  # "Last spring"
        (1, 'bank', 21): 'river_edge',  # "bank of the river"
        (1, 'crane', 51): 'bird',  # "crane wade through"
        (1, 'cell', 97): 'phone',  # "cell buzzed"
        (1, 'match', 175): 'competition',  # "tennis match"
        (1, 'bank', 226): 'financial_institution',  # "at the bank trying to sort out a loan"
        (1, 'spring', 299): 'coil',  # "broken spring in the chair"
        (1, 'cell', 381): 'phone',  # "check her cell"
        # P2
        (2, 'crane', 35): 'machine',  # "massive crane lifting"
        (2, 'bat', 186): 'animal',  # "bat swooped"
        (2, 'bats', 253): 'animal',  # "colony of bats"
        (2, 'bat', 306): 'sports_equipment',  # "baseball bat"
        (2, 'spring', 424): 'coil',  # "lost its spring"
        # P3
        (3, 'match', 11): 'fire_starter',  # "struck a match"
        (3, 'match', 52): 'fire_starter',  # "first match was damp"
        (3, 'cell', 123): 'biology',  # "cell division"
        (3, 'cell', 159): 'biology',  # "single cell splits"
        (3, 'cell', 226): 'prison',  # "prison cell"
        (3, 'spring', 298): 'coil',  # "broken spring in his mattress"
        (3, 'spring', 362): 'water_source',  # "natural spring"
        (3, 'match', 432): 'to_pair',  # "didn't match the metallic taste"
        # P4
        (4, 'match', 32): 'to_pair',  # "don't match the carpet"
        (4, 'bank', 82): 'financial_institution',  # "works at a bank"
        (4, 'bank', 110): 'financial_institution',  # "same bank"
        (4, 'spring', 152): 'water_source',  # "spring-fed pond"
        (4, 'cranes', 184): 'bird',  # "cranes gather"
        (4, 'spring', 209): 'season',  # "in the spring"
        (4, 'cell', 262): 'phone',  # "from her cell"
        (4, 'match', 308): 'to_pair',  # "pictures don't match"
        (4, 'crane', 354): 'bird',  # "see a crane take flight"
        # P5
        (5, 'bat', 38): 'animal',  # "bat echolocation"
        (5, 'bats', 59): 'animal',  # "bats navigate"
        (5, 'cell', 107): 'biology',  # "model cell"
        (5, 'spring', 121): 'coil',  # "spring-loaded"
        (5, 'cell', 165): 'biology',  # "cell membranes"
        (5, 'match', 231): 'to_pair',  # "perfect match of creativity"
        (5, 'bank', 295): 'river_edge',  # "bank of the lake"
        (5, 'match', 372): 'fire_starter',  # "light a match"
        # P6
        (6, 'match', 16): 'competition',  # "championship match"
        (6, 'bat', 79): 'sports_equipment',  # "bat swings"
        (6, 'match', 131): 'to_pair',  # "timing doesn't match"
        (6, 'spring', 196): 'coil',  # "uses a spring"
        (6, 'crane', 254): 'machine',  # "a crane has been brought in"
        (6, 'match', 323): 'to_pair',  # "didn't match safety codes"
        (6, 'bank', 369): 'river_edge',  # "bank of the creek"
        (6, 'crane', 429): 'bird',  # "spotted a crane standing"
        # P7
        (7, 'cell', 3): 'phone',  # "My cell battery"
        (7, 'bank', 99): 'financial_institution',  # "accepted by the bank"
        (7, 'match', 123): 'fire_starter',  # "borrow a match"
        (7, 'bat', 183): 'animal',  # "A bat flew past"
        (7, 'bat', 241): 'sports_equipment',  # "my son's bat"
        (7, 'spring', 295): 'season',  # "Next spring"
        (7, 'spring', 364): 'water_source',  # "natural spring"
        (7, 'cranes', 405): 'bird',  # "watching cranes"
        (7, 'matches', 451): 'to_pair',  # "actually matches"
    }

    correct = 0
    total = 0

    for occ in occurrences:
        if occ.predicted_sense:
            key = (occ.paragraph_num, occ.word, occ.start_char)
            if key in ground_truth:
                total += 1
                if occ.predicted_sense == ground_truth[key]:
                    correct += 1

    accuracy = correct / total if total > 0 else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>WSD - {method_name}</title>
    <style>
        body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.8; }}
        h1 {{ text-align: center; }}
        h2 {{ text-align: center; color: #666; }}
        .stats {{ text-align: center; font-size: 1.2em; margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 8px; }}
        .accuracy {{ font-size: 1.5em; font-weight: bold; color: {"#27ae60" if accuracy > 0.8 else "#e74c3c" if accuracy < 0.6 else "#f39c12"}; }}
        .paragraph {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }}
        .word {{ padding: 2px 6px; border-radius: 4px; font-weight: bold; cursor: help; }}
        .wrong {{ text-decoration: underline wavy red; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; padding: 15px; background: #eee; border-radius: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 14px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        .legend-section {{ margin-bottom: 10px; }}
        .legend-title {{ font-weight: bold; width: 100%; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>Word Sense Disambiguation</h1>
    <h2>{method_name}</h2>

    <div class="stats">
        Accuracy: <span class="accuracy">{correct}/{total} ({accuracy:.1%})</span>
    </div>

    <div class="legend">
"""

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

    paragraphs = story.strip().split('\n')

    for para_idx, paragraph in enumerate(paragraphs):
        para_occs = [o for o in occurrences if o.paragraph_num == para_idx + 1]
        para_occs_sorted = sorted(para_occs, key=lambda x: x.start_char, reverse=True)

        annotated = paragraph
        for occ in para_occs_sorted:
            if occ.predicted_sense:
                color = sense_colors.get(occ.predicted_sense, '#999')
                word_text = annotated[occ.start_char:occ.end_char]

                # Check if wrong
                key = (occ.paragraph_num, occ.word, occ.start_char)
                is_wrong = key in ground_truth and occ.predicted_sense != ground_truth[key]
                wrong_class = ' wrong' if is_wrong else ''

                replacement = f'<span class="word{wrong_class}" style="background: {color}; color: white;" title="{occ.predicted_sense} ({occ.confidence:.0%})">{word_text}</span>'
                annotated = annotated[:occ.start_char] + replacement + annotated[occ.end_char:]

        html += f'<div class="paragraph"><strong>Paragraph {para_idx + 1}:</strong> {annotated}</div>\n'

    html += """
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)
    print(f"  Saved: {output_file}")

    return accuracy


def main():
    from transformers import AutoModel, AutoTokenizer

    print("=" * 70)
    print("COMPARING ALL METHODS ON STORY")
    print("=" * 70)

    # Load datasets and story
    print("\nLoading data...")
    datasets = load_datasets("polysemous_datasets.json")

    with open("claudes polysem story.txt", "r") as f:
        story = f.read()

    target_words = ["bank", "bat", "crane", "cell", "spring", "match"]

    results = {}

    # ============================================================
    # Method 1: GPT-2 Basic
    # ============================================================
    print("\n" + "=" * 50)
    print("METHOD 1: GPT-2 Basic (target position)")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    probes, train_embs, train_labs = {}, {}, {}
    for word in target_words:
        print(f"  Training probe for '{word}'...")
        emb, lab = extract_training_embeddings(model, tokenizer, datasets[word], "basic")
        if emb is not None:
            probe = train_contrastive(emb, lab, output_dim=32, epochs=200, lr=0.001)
            probes[word] = probe
            train_embs[word] = emb
            train_labs[word] = lab

    occurrences = find_word_occurrences(story, target_words)
    occurrences = extract_basic(model, tokenizer, occurrences)
    occurrences = predict_senses(occurrences, probes, train_embs, train_labs, datasets)
    results['gpt2_basic'] = create_html(story, occurrences, "GPT-2 Basic (target position)", "story_gpt2_basic.html")

    # ============================================================
    # Method 2: GPT-2 Attention Flow
    # ============================================================
    print("\n" + "=" * 50)
    print("METHOD 2: GPT-2 Attention Flow")
    print("=" * 50)

    # Need eager attention for attention weights
    model = AutoModel.from_pretrained("gpt2", attn_implementation='eager')

    probes, train_embs, train_labs = {}, {}, {}
    for word in target_words:
        print(f"  Training probe for '{word}'...")
        emb, lab = extract_training_embeddings(model, tokenizer, datasets[word], "attention_flow")
        if emb is not None:
            probe = train_contrastive(emb, lab, output_dim=32, epochs=200, lr=0.001)
            probes[word] = probe
            train_embs[word] = emb
            train_labs[word] = lab

    occurrences = find_word_occurrences(story, target_words)
    occurrences = extract_attention_flow(model, tokenizer, occurrences)
    occurrences = predict_senses(occurrences, probes, train_embs, train_labs, datasets)
    results['gpt2_attn_flow'] = create_html(story, occurrences, "GPT-2 Attention Flow", "story_gpt2_attention_flow.html")

    # ============================================================
    # Method 3: GPT-2 End of Sentence
    # ============================================================
    print("\n" + "=" * 50)
    print("METHOD 3: GPT-2 End of Sentence")
    print("=" * 50)

    model = AutoModel.from_pretrained("gpt2")

    probes, train_embs, train_labs = {}, {}, {}
    for word in target_words:
        print(f"  Training probe for '{word}'...")
        emb, lab = extract_training_embeddings(model, tokenizer, datasets[word], "end_of_sentence")
        if emb is not None:
            probe = train_contrastive(emb, lab, output_dim=32, epochs=200, lr=0.001)
            probes[word] = probe
            train_embs[word] = emb
            train_labs[word] = lab

    occurrences = find_word_occurrences(story, target_words)
    occurrences = extract_end_of_sentence(model, tokenizer, occurrences)
    occurrences = predict_senses(occurrences, probes, train_embs, train_labs, datasets)
    results['gpt2_eos'] = create_html(story, occurrences, "GPT-2 End of Sentence", "story_gpt2_end_of_sentence.html")

    # ============================================================
    # Method 4: RoBERTa (bidirectional)
    # ============================================================
    print("\n" + "=" * 50)
    print("METHOD 4: RoBERTa (bidirectional)")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")

    probes, train_embs, train_labs = {}, {}, {}
    for word in target_words:
        print(f"  Training probe for '{word}'...")
        emb, lab = extract_training_embeddings(model, tokenizer, datasets[word], "basic")
        if emb is not None:
            probe = train_contrastive(emb, lab, output_dim=32, epochs=200, lr=0.001)
            probes[word] = probe
            train_embs[word] = emb
            train_labs[word] = lab

    occurrences = find_word_occurrences(story, target_words)
    occurrences = extract_basic(model, tokenizer, occurrences)
    occurrences = predict_senses(occurrences, probes, train_embs, train_labs, datasets)
    results['roberta'] = create_html(story, occurrences, "RoBERTa (bidirectional)", "story_roberta.html")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method, accuracy in results.items():
        print(f"  {method}: {accuracy:.1%}")

    print("\nOpen these files in your browser to compare:")
    print("  - story_gpt2_basic.html")
    print("  - story_gpt2_attention_flow.html")
    print("  - story_gpt2_end_of_sentence.html")
    print("  - story_roberta.html")


if __name__ == "__main__":
    main()

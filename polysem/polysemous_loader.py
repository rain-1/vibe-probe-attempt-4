"""
Polysemous Word Dataset Loader

Loads datasets for training probes to distinguish word senses.
Each dataset contains sentences where a target word is used with different meanings.
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class SenseExample:
    """A single example of a word used in context."""
    text: str
    word: str
    sense: str
    sense_id: int  # numeric label for the sense
    
    def find_word_position(self) -> Optional[int]:
        """Find the token position of the target word (approximate, assumes space tokenization)."""
        tokens = self.text.lower().split()
        word_lower = self.word.lower()
        for i, token in enumerate(tokens):
            # Handle punctuation attached to word
            clean_token = token.strip('.,!?;:"\'-')
            if clean_token == word_lower or clean_token == word_lower + 's':
                return i
        return None


@dataclass  
class PolysemousDataset:
    """Dataset for a single polysemous word."""
    word: str
    senses: list[str]
    examples: list[SenseExample]
    
    def get_sense_examples(self, sense: str) -> list[SenseExample]:
        """Get all examples for a specific sense."""
        return [ex for ex in self.examples if ex.sense == sense]
    
    def get_texts_and_labels(self) -> tuple[list[str], list[int]]:
        """Return texts and numeric labels for training."""
        texts = [ex.text for ex in self.examples]
        labels = [ex.sense_id for ex in self.examples]
        return texts, labels
    
    def train_test_split(self, test_ratio: float = 0.2, seed: int = 42):
        """Split into train/test sets, stratified by sense."""
        import random
        random.seed(seed)
        
        train_examples = []
        test_examples = []
        
        for sense in self.senses:
            sense_examples = self.get_sense_examples(sense)
            random.shuffle(sense_examples)
            split_idx = int(len(sense_examples) * (1 - test_ratio))
            train_examples.extend(sense_examples[:split_idx])
            test_examples.extend(sense_examples[split_idx:])
        
        random.shuffle(train_examples)
        random.shuffle(test_examples)
        
        return train_examples, test_examples
    
    def summary(self) -> str:
        """Print a summary of the dataset."""
        lines = [f"Word: {self.word}", f"Senses: {self.senses}", "Examples per sense:"]
        for sense in self.senses:
            count = len(self.get_sense_examples(sense))
            lines.append(f"  {sense}: {count}")
        return "\n".join(lines)


def load_datasets(path: str = "polysemous_datasets.json") -> dict[str, PolysemousDataset]:
    """Load all polysemous word datasets from JSON file."""
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
        
        datasets[word] = PolysemousDataset(
            word=word,
            senses=senses,
            examples=examples
        )
    
    return datasets


def print_all_summaries(datasets: dict[str, PolysemousDataset]):
    """Print summaries for all datasets."""
    for word, dataset in datasets.items():
        print(dataset.summary())
        print("-" * 40)


# Example usage
if __name__ == "__main__":
    datasets = load_datasets()
    print_all_summaries(datasets)
    
    # Example: get data ready for training
    print("\n\nExample: Preparing 'bank' for training:")
    bank = datasets["bank"]
    train, test = bank.train_test_split()
    print(f"Train examples: {len(train)}")
    print(f"Test examples: {len(test)}")
    
    print("\nSample training examples:")
    for ex in train[:3]:
        print(f"  [{ex.sense}] {ex.text}")

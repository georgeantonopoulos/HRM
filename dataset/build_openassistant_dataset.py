from typing import Optional, List, Dict, Tuple
import os
import json
import numpy as np
import string

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from datasets import load_dataset

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "OpenAssistant/oasst1"
    output_dir: str = "data/openassistant-oasst1"
    
    max_seq_len: int = 512
    subsample_size: Optional[int] = None
    min_text_length: int = 10
    max_text_length: int = 2048
    
    # Character vocabulary settings
    include_special_chars: bool = True


def build_char_vocab(include_special: bool = True) -> Dict[str, int]:
    """Build character vocabulary mapping"""
    # Start with PAD token
    vocab = {'<PAD>': 0}
    
    # Add basic ASCII characters
    chars = []
    
    # Letters and digits
    chars.extend(string.ascii_letters + string.digits)
    
    # Common punctuation and symbols
    chars.extend(' .,!?;:\'"()-[]{}/@#$%^&*+=_`~\n\t')
    
    if include_special:
        # Add more special characters that might appear in conversations
        chars.extend('|\\<>«»""''–—…°±×÷√π∞≤≥≠')
    
    # Remove duplicates and sort for consistency
    chars = sorted(set(chars))
    
    # Add to vocabulary
    for i, char in enumerate(chars, 1):
        vocab[char] = i
        
    # Add unknown token
    vocab['<UNK>'] = len(vocab)
    
    return vocab


def text_to_sequence(text: str, char_vocab: Dict[str, int], max_len: int) -> np.ndarray:
    """Convert text to sequence of character IDs"""
    # Truncate if too long
    if len(text) > max_len:
        text = text[:max_len]
    
    # Convert to character IDs
    sequence = []
    for char in text:
        if char in char_vocab:
            sequence.append(char_vocab[char])
        else:
            sequence.append(char_vocab['<UNK>'])
    
    # Pad to max_len
    while len(sequence) < max_len:
        sequence.append(char_vocab['<PAD>'])
        
    return np.array(sequence, dtype=np.int32)


def extract_conversation_pairs(dataset) -> List[Tuple[str, str]]:
    """Extract (prompter, assistant) pairs from the dataset"""
    pairs = []
    
    # Group messages by conversation tree
    trees = {}
    for item in dataset:
        tree_id = item['message_tree_id']
        if tree_id not in trees:
            trees[tree_id] = []
        trees[tree_id].append(item)
    
    # Extract pairs from each tree
    for tree_id, messages in trees.items():
        # Build parent-child mapping
        children = {}
        for msg in messages:
            parent_id = msg['parent_id']
            if parent_id is not None:
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(msg)
        
        # Find conversation pairs
        for msg in messages:
            if msg['role'] == 'prompter' and msg['message_id'] in children:
                prompter_text = msg['text']
                
                # Find assistant responses
                for child in children[msg['message_id']]:
                    if child['role'] == 'assistant':
                        assistant_text = child['text']
                        pairs.append((prompter_text, assistant_text))
    
    return pairs


def convert_subset(set_name: str, config: DataProcessConfig, char_vocab: Dict[str, int]) -> Dict:
    """Convert a dataset subset to the required format"""
    print(f"Loading {set_name} split...")
    
    # Load dataset
    dataset = load_dataset(config.source_repo, split=set_name)
    
    # Extract conversation pairs
    print(f"Extracting conversation pairs from {len(dataset)} messages...")
    pairs = extract_conversation_pairs(dataset)
    
    # Filter by text length
    filtered_pairs = []
    for prompt, response in pairs:
        if (config.min_text_length <= len(prompt) <= config.max_text_length and 
            config.min_text_length <= len(response) <= config.max_text_length):
            filtered_pairs.append((prompt, response))
    
    pairs = filtered_pairs
    print(f"After filtering: {len(pairs)} conversation pairs")
    
    # Subsample if requested
    if set_name == "train" and config.subsample_size is not None:
        if config.subsample_size < len(pairs):
            indices = np.random.choice(len(pairs), size=config.subsample_size, replace=False)
            pairs = [pairs[i] for i in indices]
            print(f"Subsampled to {len(pairs)} pairs")
    
    # Convert to sequences
    print("Converting text to sequences...")
    inputs = []
    labels = []
    
    for prompt, response in tqdm(pairs):
        input_seq = text_to_sequence(prompt, char_vocab, config.max_seq_len)
        label_seq = text_to_sequence(response, char_vocab, config.max_seq_len)
        
        inputs.append(input_seq)
        labels.append(label_seq)
    
    # Create dataset structure matching other builders
    results = {
        "inputs": np.stack(inputs, 0),
        "labels": np.stack(labels, 0),
        "puzzle_identifiers": np.zeros(len(pairs), dtype=np.int32),  # All same type
        "puzzle_indices": np.arange(len(pairs) + 1, dtype=np.int32),  # Each pair is separate
        "group_indices": np.arange(len(pairs) + 1, dtype=np.int32)    # Each pair is separate group
    }
    
    return results


def convert_dataset(config: DataProcessConfig):
    """Convert the full OpenAssistant dataset"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Build character vocabulary
    print("Building character vocabulary...")
    char_vocab = build_char_vocab(config.include_special_chars)
    print(f"Vocabulary size: {len(char_vocab)}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save character vocabulary
    with open(os.path.join(config.output_dir, "char_vocab.json"), "w") as f:
        json.dump(char_vocab, f, indent=2)
    
    # Process each split
    splits_data = {}
    for split_name in ["train", "validation"]:
        print(f"\nProcessing {split_name} split...")
        
        try:
            split_data = convert_subset(split_name, config, char_vocab)
            splits_data[split_name] = split_data
            
            # Save split data
            split_dir = os.path.join(config.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for key, value in split_data.items():
                np.save(os.path.join(split_dir, f"all__{key}.npy"), value)
            
            # Create metadata
            metadata = PuzzleDatasetMetadata(
                seq_len=config.max_seq_len,
                vocab_size=len(char_vocab),
                
                pad_id=char_vocab['<PAD>'],
                ignore_label_id=char_vocab['<PAD>'],
                
                blank_identifier_id=0,
                num_puzzle_identifiers=1,
                
                total_groups=len(split_data["group_indices"]) - 1,
                mean_puzzle_examples=1.0,
                sets=["all"]
            )
            
            # Save metadata
            with open(os.path.join(split_dir, "dataset.json"), "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)
                
            print(f"Saved {split_name}: {len(split_data['inputs'])} examples")
            
        except Exception as e:
            print(f"Warning: Could not process {split_name} split: {e}")
    
    # Save global identifiers mapping (for compatibility)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<conversation>"], f)
    
    print(f"\nDataset conversion complete! Saved to {config.output_dir}")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
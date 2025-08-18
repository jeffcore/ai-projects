import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

class TextDataset(Dataset):
    """
    Dataset for training the transformer model on text data.
    """
    def __init__(self, data, tokenizer, seq_length):
        """
        Initialize the dataset.
        
        Args:
            data: List of text samples
            tokenizer: Tokenizer to encode the text
            seq_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Encode all text samples
        self.encoded_data = []
        for text in data:
            self.encoded_data.append(tokenizer.encode(text))
            
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        # Get encoded text
        tokens = self.encoded_data[idx]
        
        # If the sequence is too long, truncate it
        if len(tokens) > self.seq_length:
            # Randomly select a starting point
            start_idx = random.randint(0, len(tokens) - self.seq_length)
            tokens = tokens[start_idx:start_idx + self.seq_length]
        
        # If the sequence is too short, pad it
        if len(tokens) < self.seq_length:
            # Pad with the pad token ID
            pad_token_id = self.tokenizer.get_pad_token_id()
            tokens = tokens + [pad_token_id] * (self.seq_length - len(tokens))
        
        # Convert to tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }

def load_text_data(data_dir, file_extension=".txt"):
    """
    Load text data from a directory.
    
    Args:
        data_dir: Directory containing text files
        file_extension: File extension to look for
        
    Returns:
        List of text samples
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(file_extension):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                data.append(text)
    return data

def create_dataloader(dataset, batch_size, shuffle=True):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create a DataLoader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

class WikiTextDataset(Dataset):
    """
    Dataset for training the transformer model on WikiText data.
    This is a more efficient implementation for larger datasets.
    """
    def __init__(self, data_path, tokenizer, seq_length):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file
            tokenizer: Tokenizer to encode the text
            seq_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load and tokenize the entire dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        
        # Calculate the number of samples
        self.num_samples = len(self.tokens) // seq_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Calculate start and end indices
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 to include the target for the last token
        
        # Get the tokens for this sample
        tokens = self.tokens[start_idx:end_idx]
        
        # If we don't have enough tokens, pad
        if len(tokens) < self.seq_length + 1:
            pad_token_id = self.tokenizer.get_pad_token_id()
            tokens = tokens + [pad_token_id] * ((self.seq_length + 1) - len(tokens))
        
        # Create input and target tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }

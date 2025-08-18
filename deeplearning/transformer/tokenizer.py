from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

class SimpleTokenizer:
    """
    A simple BPE tokenizer for the transformer model.
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", 0),
                ("[EOS]", 1),
                ("[UNK]", 2),
                ("[PAD]", 3),
            ],
        )
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[BOS]", "[EOS]", "[UNK]", "[PAD]"],
            min_frequency=2
        )
        
    def train(self, files):
        """
        Train the tokenizer on a list of files.
        
        Args:
            files: List of file paths to train on
        """
        self.tokenizer.train(files, self.trainer)
        
    def save(self, path):
        """
        Save the tokenizer to a file.
        
        Args:
            path: Path to save the tokenizer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        
    def load(self, path):
        """
        Load a tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
        """
        self.tokenizer = Tokenizer.from_file(path)
        
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        """
        Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self.tokenizer.get_vocab_size()
    
    def get_pad_token_id(self):
        """
        Get the ID of the padding token.
        
        Returns:
            Padding token ID
        """
        return self.tokenizer.token_to_id("[PAD]")
    
    def get_bos_token_id(self):
        """
        Get the ID of the beginning of sequence token.
        
        Returns:
            BOS token ID
        """
        return self.tokenizer.token_to_id("[BOS]")
    
    def get_eos_token_id(self):
        """
        Get the ID of the end of sequence token.
        
        Returns:
            EOS token ID
        """
        return self.tokenizer.token_to_id("[EOS]")

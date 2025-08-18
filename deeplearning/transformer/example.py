import torch
from transformer_model import DecoderOnlyTransformer

def main():
    # Define model parameters
    vocab_size = 10000  # Example vocabulary size
    d_model = 256       # Embedding dimension
    num_layers = 4      # Number of decoder layers
    num_heads = 8       # Number of attention heads
    d_ff = 1024         # Feed-forward dimension
    max_seq_length = 100  # Maximum sequence length
    
    # Initialize the model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=0.1
    )
    
    # Print model summary
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example input (batch of token IDs)
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Example of text generation
    print("\nGenerating text from seed input...")
    generated = model.generate(input_ids, max_length=20, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")
    
    # In a real application, you would convert these token IDs back to text
    # using a tokenizer's decode method
    
if __name__ == "__main__":
    main()

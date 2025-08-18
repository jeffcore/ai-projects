import torch
import argparse
import os

from transformer_model import DecoderOnlyTransformer
from tokenizer import SimpleTokenizer

def load_model(checkpoint_path, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with the same parameters
    model = DecoderOnlyTransformer(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        d_ff=checkpoint['d_ff'],
        max_seq_length=checkpoint['seq_length'],
        dropout=checkpoint['dropout']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def generate_text(model, tokenizer, prompt, max_length, temperature, device):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to run the model on
        
    Returns:
        Generated text
    """
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=max_length, temperature=temperature)
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained transformer model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer file")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Determine device (use MPS for M3 Max if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(args.tokenizer)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Generate text
    for i in range(args.num_samples):
        print(f"\nSample {i+1}:")
        generated_text = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            args.max_length, 
            args.temperature, 
            device
        )
        print(generated_text)

if __name__ == "__main__":
    main()

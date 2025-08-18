import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformer_model import DecoderOnlyTransformer
from tokenizer import SimpleTokenizer
from dataset import WikiTextDataset, create_dataloader
from download_data import download_wikitext

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    # Set random seed
    set_seed(args.seed)
    
    # Determine device (use MPS for M3 Max if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Create TensorBoard log directory
    log_dir = os.path.join(args.log_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download dataset if it doesn't exist
    data_file = os.path.join(args.data_dir, f"{args.subset}-{args.split}.txt")
    if not os.path.exists(data_file):
        data_file = download_wikitext(args.data_dir, args.dataset, args.subset, args.split)
    
    # Create tokenizer directory if it doesn't exist
    os.makedirs(args.tokenizer_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer_path = os.path.join(args.tokenizer_dir, f"{args.subset}-tokenizer.json")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    
    # Train or load tokenizer
    if not os.path.exists(tokenizer_path):
        print("Training tokenizer...")
        tokenizer.train([data_file])
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    
    # Get actual vocabulary size after tokenizer training
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = WikiTextDataset(data_file, tokenizer, args.seq_length)
    dataloader = create_dataloader(dataset, args.batch_size, shuffle=True)
    
    # Initialize model
    print("Initializing model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.seq_length,
        dropout=args.dropout
    )
    model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    # Log model architecture
    dummy_input = torch.zeros(1, args.seq_length, dtype=torch.long).to(device)
    writer.add_graph(model, dummy_input)
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics to TensorBoard
        writer.add_scalar('training/loss', avg_loss, epoch + 1)
        writer.add_scalar('training/learning_rate', scheduler.get_last_lr()[0], epoch + 1)
        writer.add_scalar('training/epoch_time', epoch_time, epoch + 1)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Save checkpoint if it's the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab_size': vocab_size,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'd_ff': args.d_ff,
                'seq_length': args.seq_length,
                'dropout': args.dropout,
            }, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': vocab_size,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'd_ff': args.d_ff,
                'seq_length': args.seq_length,
                'dropout': args.dropout,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Generate sample text
        if (epoch + 1) % args.generate_every == 0:
            sample_text = generate_sample_text(model, tokenizer, device, args.seq_length)
            writer.add_text('generated_text', sample_text, epoch + 1)

def generate_sample_text(model, tokenizer, device, seq_length, temperature=0.8, max_length=100):
    """Generate sample text from the model."""
    model.eval()
    
    # Start with a BOS token
    input_ids = torch.tensor([[tokenizer.get_bos_token_id()]], dtype=torch.long).to(device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=max_length, temperature=temperature)
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print("\nGenerated sample:")
    print(generated_text)
    print()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Train a decoder-only transformer model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--subset", type=str, default="wikitext-2-v1", help="Dataset subset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store TensorBoard logs")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory to store tokenizer")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--generate_every", type=int, default=1, help="Generate sample text every N epochs")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()

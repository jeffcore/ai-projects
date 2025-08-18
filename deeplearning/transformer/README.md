# Decoder-Only Transformer Implementation in PyTorch

This repository contains a simple implementation of a decoder-only transformer model in PyTorch, similar to the architecture used in models like GPT.

## Files

- `transformer_model.py`: Contains the implementation of the decoder-only transformer model
- `example.py`: Demonstrates how to use the model for inference and text generation
- `requirements.txt`: Lists the dependencies required to run the code

## Model Architecture

The implementation includes the following components:

- **PositionalEncoding**: Adds positional information to the input embeddings
- **MultiHeadAttention**: Implements the multi-head self-attention mechanism
- **FeedForward**: Position-wise feed-forward network
- **DecoderLayer**: Combines self-attention and feed-forward networks with residual connections and layer normalization
- **DecoderOnlyTransformer**: The main model that stacks multiple decoder layers

## Usage

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the example script:

   ```bash
   python example.py
   ```

## Training the Model

1. Download the dataset (WikiText-2 by default):

   ```bash
   python download_data.py
   ```

2. Train the model:

   ```bash
   python train.py
   ```

3. Generate text with the trained model:

   ```bash
   python generate.py --checkpoint checkpoints/model_epoch_10.pt --tokenizer tokenizer/wikitext-2-v1-tokenizer.json --prompt "Once upon a time"
   ```

## Customization

You can customize the model by adjusting the following parameters:

- `vocab_size`: Size of the vocabulary
- `d_model`: Dimension of the model (embedding dimension)
- `num_layers`: Number of decoder layers
- `num_heads`: Number of attention heads
- `d_ff`: Dimension of the feed-forward network
- `max_seq_length`: Maximum sequence length
- `dropout`: Dropout rate for regularization

## Text Generation

The model includes a `generate` method that performs autoregressive text generation using the trained model. You can control the generation process with the `temperature` parameter:

- `temperature < 1.0`: More deterministic outputs
- `temperature = 1.0`: Standard sampling
- `temperature > 1.0`: More random outputs

# demo.py
# -----------------------------------------------------------------------------
# NanoDecoder (GPT-style Decoder) Demo Script
# -----------------------------------------------------------------------------
# This script demonstrates how to:
# 1. Prepare a toy text dataset and create a simple word-level tokenizer.
# 2. Sample training batches for next-token prediction.
# 3. Define the NanoGPT model using stacked Transformer blocks.
# 4. Train the model using Cross-Entropy loss and AdamW optimizer.
# 5. Generate new text autoregressively starting from a prompt token.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Import the Transformer Block definition from transformer_block.py
from transformer_block import Block


# -----------------------------------------------------------------------------
# 1. Environment & Hardware Diagnostics
# -----------------------------------------------------------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# -----------------------------------------------------------------------------
# 2. Text Corpus & Tokenization (Word-Level Vocabulary)
# -----------------------------------------------------------------------------
# Toy text dataset containing short sentences on various themes
corpus = [
    # Greetings
    "hello friends how are you",
    "hi friends how are you doing",
    "hello everyone hope you are doing well",
    "hey friends how is everyone",
    "good morning friends how are you",
    
    # Tea / temperature
    "the tea is very hot",
    "this tea is extremely hot",
    "the tea is too hot to drink",
    "my cup of tea is really hot",
    "the tea is steaming hot",
    
    # Personal information
    "my name is Aarohi",
    "i am Aarohi",
    "Aarohi is my name",
    "people call me Aarohi",
    "my name is Aarohi and i live in India",
    
    # Delhi / roads
    "the roads of Delhi are busy",
    "Delhi roads are crowded",
    "the streets of Delhi are very busy",
    "traffic is heavy on the roads of Delhi",
    "Delhi has crowded and busy roads",
    
    # Mumbai / rain
    "it is raining in Mumbai",
    "Mumbai is experiencing heavy rain",
    "it is raining heavily in Mumbai",
    "Mumbai has received a lot of rain",
    "the weather in Mumbai is rainy today",
    
    # Train
    "the train is late again",
    "the train has been delayed again",
    "my train is running late",
    "the train arrived late once again",
    "the train is delayed today",
    
    # Samosas and tea
    "i love eating samosas and drinking tea",
    "i enjoy samosas with a cup of tea",
    "samosas and tea are my favorite combination",
    "i like eating hot samosas and drinking tea",
    "i really enjoy having samosas with tea",
    
    # Holi
    "holi is my favorite festival",
    "i love celebrating Holi",
    "Holi is the festival i enjoy the most",
    "my favorite Indian festival is Holi",
    "i really like the colors and celebrations of Holi",
    
    # Diwali
    "diwali brings lights and sweets",
    "Diwali is celebrated with lights and sweets",
    "diwali fills homes with lights and sweets",
    "people celebrate Diwali with lamps and sweets",
    "Diwali is a festival of lights and delicious sweets",
    
    # Cricket
    "india won the cricket match",
    "India has won the cricket game",
    "Sachin is the god of Cricket",
    "Sachin Tendulkar is known as the god of cricket",
    "many fans consider Sachin the god of cricket",
    "Sachin is regarded as a cricket legend",
    "Sachin is one of the greatest cricket players ever",
    "people call Sachin the god of cricket",
    "Sachin Tendulkar is a legendary cricketer",
    "Sachin has earned the title of the god of cricket",
    "cricket fans admire Sachin as a great player",
    "Sachin is considered a cricketing icon"
]

# Append special "<END>" token to mark end of each sentence in corpus
corpus = [s + " <END>" for s in corpus]

# Combine all sentences into one long string separated by space
text = " ".join(corpus)

# Extract unique words to form the vocabulary
words = list(set(text.split()))
vocab_size = len(words)

# Create mapping dictionaries: word-to-integer ID and integer ID-to-word
word2num = {w: i for i, w in enumerate(words)}
num2word = {i: w for w, i in word2num.items()}

# Encode full text corpus into a 1D PyTorch tensor of token integer IDs
data = torch.tensor([word2num[w] for w in text.split()], dtype=torch.long)


# -----------------------------------------------------------------------------
# 3. Model Hyperparameters
# -----------------------------------------------------------------------------
block_size = 10     # Maximum context window length (number of tokens model considers at once)
embedding_dim = 32  # Feature vector dimension for each token
n_heads = 3         # Number of parallel self-attention heads
n_layers = 3        # Number of stacked Transformer blocks
lr = 1e-3           # Learning rate for AdamW optimizer
epochs = 1500       # Total training iterations


# -----------------------------------------------------------------------------
# 4. Data Batching Helper
# -----------------------------------------------------------------------------
def get_batch(batch_size=16):
    """
    Generates a random batch of inputs (x) and targets (y) for training.
    For input sequence x[i : i + block_size], the target y[i + 1 : i + block_size + 1]
    contains the next token to predict at every position.
    """
    # Pick random starting indices from the encoded dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract input context sequences of length block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    
    # Extract target sequences shifted by 1 token to the right
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    
    return x, y


# -----------------------------------------------------------------------------
# 5. NanoGPT Language Model Class
# -----------------------------------------------------------------------------
class NanoGPT(nn.Module):
    """
    Decoder-only Autoregressive Transformer Language Model.
    Architecture:
    1. Token Embedding + Positional Embedding
    2. Stacked Transformer Blocks (Multi-Head Attention + FeedForward)
    3. Final LayerNorm
    4. Language Model Linear Head (Outputs vocabulary logits)
    """
    def __init__(self):
        super().__init__()
        # Token Embedding table: maps token IDs -> vectors of size embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        # Learned Positional Embedding table: maps positions (0..block_size-1) -> vectors
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 

        # Stack of n_layers Transformer Blocks
        self.blocks = nn.Sequential(*[
            Block(embedding_dim, block_size, n_heads) 
            for _ in range(n_layers)
        ]) 

        # Final Layer Normalization before prediction head
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Output projection layer: maps embedding_dim -> vocab_size logits
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        # idx shape: (Batch size B, Sequence length T)
        B, T = idx.shape 
        
        # 1. Fetch token embeddings for each token in sequence: (B, T, embedding_dim)
        tok_emb = self.token_embedding(idx) 
        
        # 2. Fetch position embeddings for position indices 0..T-1: (T, embedding_dim)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        
        # 3. Combine token identity and positional information
        x = tok_emb + pos_emb  # Broadcasting shapes (B, T, C) + (T, C) -> (B, T, C)
        
        # 4. Pass feature vectors through the stack of Transformer blocks
        x = self.blocks(x) 
        
        # 5. Normalize features and compute output logits for each token position
        x = self.ln_f(x)
        logits = self.head(x) # Shape: (B, T, vocab_size)

        # 6. If targets are provided, compute Cross-Entropy classification loss
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            # Reshape logits to (B*T, vocab_size) and targets to (B*T,) for PyTorch cross_entropy
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T)) 
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generates new tokens given a context sequence idx (shape B, T).
        Appends predictions to context one token at a time up to max_new_tokens.
        """
        for _ in range(max_new_tokens):
            # Crop current context to max block_size if sequence exceeds window size
            idx_cond = idx[:, -block_size:]
            
            # Forward pass to get model predictions (logits)
            logits, _ = self(idx_cond)
            
            # Focus only on the logits of the last token position
            logits = logits[:, -1, :] # Shape: (B, vocab_size)
            
            # Convert logits to probability distribution via Softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token index from probability distribution
            next_idx = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)
            
            # Concatenate newly predicted token to context sequence
            idx = torch.cat((idx, next_idx), dim=1)
            
        return idx


# -----------------------------------------------------------------------------
# 6. Model Training Loop
# -----------------------------------------------------------------------------
# Instantiate model and optimizer
model = NanoGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("\nStarting model training...")
for step in range(epochs):
    # Fetch random training batch
    xb, yb = get_batch() 
    
    # Compute forward pass and loss
    logits, loss = model(xb, yb)
    
    # Backpropagation and optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Progress logging every 300 steps
    if step % 300 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")


# -----------------------------------------------------------------------------
# 7. Text Generation Demo (Trained Model)
# -----------------------------------------------------------------------------
# Start generation with prompt token "hello"
context = torch.tensor([[word2num["hello"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

# Convert predicted token IDs back to human-readable words
generated_text = " ".join(num2word[int(i)] for i in out[0])

print("\nGenerated Text (Trained Model):\n")
print(generated_text)


# -----------------------------------------------------------------------------
# 8. Saving and Loading the Model
# -----------------------------------------------------------------------------
# PyTorch recommends saving only the model's learned parameters (state_dict).
MODEL_PATH = "nano_gpt_weights.pth"
CHECKPOINT_PATH = "nano_gpt_checkpoint.pt"

# -----------------------------------------------------------------------------
# Method A: Save and Load Model Weights (state_dict)
# -----------------------------------------------------------------------------
print(f"\n--- Method A: Saving model weights to '{MODEL_PATH}' ---")
torch.save(model.state_dict(), MODEL_PATH)

print(f"Loading model weights from '{MODEL_PATH}' into a fresh NanoGPT instance...")
# 1. Create a fresh model instance with the same architecture
loaded_model = NanoGPT()

# 2. Load the state dictionary of weights into the new model instance
loaded_model.load_state_dict(torch.load(MODEL_PATH))

# 3. Set the model to evaluation mode
loaded_model.eval()

# 4. Generate text from the loaded model to verify functionality
out_loaded = loaded_model.generate(context, max_new_tokens=15)
text_loaded = " ".join(num2word[int(i)] for i in out_loaded[0])
print("\nGenerated Text (Loaded Model Weights):\n", text_loaded)


# -----------------------------------------------------------------------------
# Method B: Save and Load Full Checkpoint (Weights + Vocab + Optimizer)
# -----------------------------------------------------------------------------
print(f"\n--- Method B: Saving full checkpoint to '{CHECKPOINT_PATH}' ---")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'word2num': word2num,
    'num2word': num2word,
    'vocab_size': vocab_size,
    'block_size': block_size,
    'embedding_dim': embedding_dim,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'last_epoch': epochs,
    'last_loss': loss.item(),
}
torch.save(checkpoint, CHECKPOINT_PATH)

print(f"Loading checkpoint from '{CHECKPOINT_PATH}'...")
checkpoint_data = torch.load(CHECKPOINT_PATH)

# Re-create model using checkpoint state
checkpoint_model = NanoGPT()
checkpoint_model.load_state_dict(checkpoint_data['model_state_dict'])
checkpoint_model.eval()

out_checkpoint = checkpoint_model.generate(context, max_new_tokens=15)
text_checkpoint = " ".join(num2word[int(i)] for i in out_checkpoint[0])
print("\nGenerated Text (Loaded Full Checkpoint):\n", text_checkpoint)


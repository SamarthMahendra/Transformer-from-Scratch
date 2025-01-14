import torch
import torch.nn as nn
import numpy as np
from torch import optim
import random


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


import torch
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.relative_positions = nn.Parameter(torch.randn(max_seq_len, max_seq_len, d_model))

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.relative_positions[:seq_len, :seq_len]
        return x

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_mask=None):
        if x.size(0) > self.positional_encoding.pe.size(0):
            x = x[:self.positional_encoding.pe.size(0), :]

        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Generate the mask based on the current sequence length of x
        seq_len = x.size(0)
        src_mask = generate_causal_mask(seq_len).to(x.device)

        x = self.transformer(x, src_mask)
        logits = self.output_layer(x)
        return logits


def generate_causal_mask(seq_len):
    # Generate a causal mask with the exact sequence length
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask



class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, vocab_index, d_model=128, num_layers=4, num_heads=2, d_ff=256, max_seq_len=20):
        super().__init__()
        self.model = TransformerLanguageModel(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
        self.vocab_size = vocab_size
        self.vocab_index = vocab_index  # Store vocab_index as an attribute
        self.model.eval()  # Start in eval mode

    def get_next_char_log_probs(self, context):
        self.model.eval()

        if not context:
            # Return uniform log probs if context is empty
            return np.log(np.ones(self.vocab_size) / self.vocab_size)

        context_indices = torch.LongTensor([self.vocab_index.index_of(c) for c in context]).unsqueeze(1)

        # Truncate context to the last max_seq_len tokens
        if context_indices.size(0) > 20:
            context_indices = context_indices[-20:]

        mask = generate_causal_mask(context_indices.size(0)).to(context_indices.device)

        with torch.no_grad():
            logits = self.model(context_indices, mask)
            log_probs = nn.functional.log_softmax(logits[-1, 0], dim=0)

        # Clamp values to avoid extreme log probabilities
        min_log_prob = np.log(1e-10)  # Lower bound to avoid -inf values
        return np.maximum(log_probs.cpu().numpy(), min_log_prob)

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        total_log_prob = 0.0

        for next_char in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            char_index = self.vocab_index.index_of(next_char)

            # Check if the character exists in the vocabulary
            if char_index >= 0 and char_index < self.vocab_size:
                total_log_prob += log_probs[char_index]
            else:
                # If character is out of bounds, assign a low log probability
                total_log_prob += np.log(1e-10)

            context += next_char  # Update context with the predicted char
        return total_log_prob


import torch
import torch.nn as nn
import numpy as np
from torch import optim


def train_lm(args, train_text, dev_text, vocab_index):
    # Define model hyperparameters
    vocab_size = len(vocab_index)
    d_model = 64  # Significantly reduced hidden dimension for CPU efficiency
    num_layers = 4  # Reduced to 2 Transformer layers
    num_heads = 2  # Keep number of heads low for efficiency
    d_ff = 128  # Smaller feedforward network dimension
    max_seq_len = 20  # Shorter sequence length for faster computation
    dropout = 0.2  # Increased dropout for better generalization on limited data

    # Training hyperparameters
    learning_rate = 1e-3  # Slightly higher learning rate for faster convergence
    epochs = 4  # Fewer epochs; use early stopping if validation doesn't improve
    batch_size = 16  # Smaller batch size for CPU memory constraints

    # Initialize the language model with weight tying
    model = NeuralLanguageModel(vocab_size, vocab_index, d_model=d_model, num_layers=num_layers,
                                num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len)
    model.model.train()
    model.model.output_layer.weight = model.model.embedding.weight  # Tie weights

    optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Step decay after every 3 epochs
    loss_fcn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_tokens = 0  # To track the total number of tokens processed

        for i in range(0, len(train_text) - max_seq_len, batch_size):
            batch_texts = [train_text[j:j + max_seq_len + 1] for j in range(i, i + batch_size)]
            batch_inputs = []
            batch_targets = []

            # Prepare input and target sequences for each batch
            for text in batch_texts:
                if len(text) == max_seq_len + 1:
                    input_seq = torch.LongTensor([vocab_index.index_of(c) for c in text[:-1]])
                    target_seq = torch.LongTensor([vocab_index.index_of(c) for c in text[1:]])
                    batch_inputs.append(input_seq)
                    batch_targets.append(target_seq)

            if batch_inputs:
                batch_inputs = torch.stack(batch_inputs).transpose(0, 1)  # shape: [seq_len, batch_size]
                batch_targets = torch.stack(batch_targets).transpose(0, 1)  # shape: [seq_len, batch_size]

                mask = generate_causal_mask(max_seq_len).to(batch_inputs.device)

                optimizer.zero_grad()
                logits = model.model(batch_inputs, mask)
                loss = loss_fcn(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_targets.numel()  # Scale loss by number of tokens in batch
                num_tokens += batch_targets.numel()  # Count the total number of tokens processed

        scheduler.step()  # Update learning rate

        # Calculate average loss per token and perplexity
        avg_loss = total_loss / num_tokens
        perplexity = math.exp(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    model.model.eval()
    return model
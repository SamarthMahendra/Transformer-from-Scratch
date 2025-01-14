# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from torch.nn import functional as F
import math


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        # Add batch dimension if not present
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)

        x = self.embedding(indices)
        x = self.positional_encoding(x)
        attention_maps = []

        for layer in self.transformer_layers:
            x, attn_map = layer(x)
            attention_maps.append(attn_map)

        logits = self.output_layer(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()
        self.attention_linear_q = nn.Linear(d_model, d_model)  # Changed from d_internal
        self.attention_linear_k = nn.Linear(d_model, d_model)
        self.attention_linear_v = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Multi-head attention
        q = self.attention_linear_q(x)
        k = self.attention_linear_k(x)
        v = self.attention_linear_v(x)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # First residual connection and normalization
        x = self.norm1(x + self.dropout(self.output_projection(attn_output)))

        # Feedforward network
        ffn_output = self.ffn(x)

        # Second residual connection and normalization
        x = self.norm2(x + self.dropout(ffn_output))

        return x, attn_weights





# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 100):
        super().__init__()
        pe = torch.zeros(num_positions, d_model)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # Get the true vocabulary size from the training data
    # Find the maximum index used in any input sequence
    vocab_size = max(max(ex.input_indexed) for ex in train) + 1

    # Initialize model with correct vocabulary size
    model = Transformer(vocab_size=vocab_size,
                        num_positions=20,  # As specified in the original code
                        d_model=128,
                        d_internal=64,
                        num_classes=3,
                        num_layers=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    batch_size = 32  # Added batch processing for efficiency

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train)
        total_loss = 0.0

        # Process in batches
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]

            # Zero out gradients
            optimizer.zero_grad()

            batch_loss = 0
            for example in batch:
                # Forward pass
                log_probs, _ = model(example.input_tensor)

                # Calculate loss
                loss = loss_fcn(log_probs, example.output_tensor)
                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(batch)

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss * batch_size / len(train)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Evaluate on dev set
        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            correct = 0
            total = 0

            for example in dev:
                log_probs, _ = model(example.input_tensor)
                dev_loss += loss_fcn(log_probs, example.output_tensor).item()

                predictions = torch.argmax(log_probs, dim=1)
                correct += (predictions == example.output_tensor).sum().item()
                total += len(example.output_tensor)

            dev_accuracy = correct / total
            print(f"Dev Accuracy: {dev_accuracy:.4f}")

    return model



####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

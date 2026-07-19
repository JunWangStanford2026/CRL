import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

class MiniLM(nn.Module):
    '''
        embedding (w/ positional encodings) -> transformer blocks -> linear layer -> softmax
        outputs probabilities in [0, 1], shape (1, N)
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MiniLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


        # Sinusoidal Position Encoding
        pe = torch.zeros(512, embedding_dim)
        position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)          # (512, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )                                                                          # (embedding_dim/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        self.register_buffer('positional_encoding', pe.unsqueeze(0))             # (1, 512, embedding_dim)


        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        '''
        x has shape (B, 2N)
        '''
        B = x.size(0)
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        transformer_output = self.transformer_encoder(embedded.transpose(0, 1)).transpose(0, 1)  # shape (B, 2N, embedding_dim)
        assert transformer_output.shape == (B, x.size(1), self.embedding_dim)
        transformer_output = transformer_output + embedded  # residual connection
        output = self.linear(transformer_output[:, -1, :])  # take the last token's output, shape (B, vocab_size)
        return torch.softmax(output, dim=-1)



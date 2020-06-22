import torch
import torch.nn as nn
import torch.nn.functional as F

def find_lengths(messages: torch.Tensor) -> torch.Tensor:
    max_k = messages.size(1)
    zero_mask = messages == 0

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p = 0.7)

        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        return self.lin2(self.act(self.norm(self.dropout(self.lin1(inputs)))))

"""
the functions below are credited to https://github.com/facebookresearch/egg
"""

class RnnEncoder(nn.Module):
    """Feeds a sequence into an RNN (vanilla RNN, GRU, LSTM) cell and returns a vector representation 
    of it, which is found as the last hidden state of the last RNN layer. Assumes that the eos token has the id equal to 0.
    """

    def __init__(self, vocab_size, embedding, n_hidden, cell='rnn', num_layers=1):
        """
        Arguments:
            vocab_size {int} -- The size of the input vocabulary (including eos)
            emb {nn.Embedding} -- Dimensionality of the embeddings
            n_hidden {int} -- Dimensionality of the cell's hidden state
        
        Keyword Arguments:
            cell {str} -- Type of the cell ('rnn', 'gru', or 'lstm') (default: {'rnn'})
            num_layers {int} -- Number of the stacked RNN layers (default: {1})
        """
        super(RnnEncoder, self).__init__()

        
        self.embedding = embedding
        self.embed_dim = embedding.embedding_dim

        cell = cell.lower()
        cell_types = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.cell = cell_types[cell](input_size=self.embed_dim, batch_first=True,
                               hidden_size=n_hidden, num_layers=num_layers)


    def forward(self, message, lengths=None):
        """Feeds a sequence into an RNN cell and returns the last hidden state of the last layer.
        Arguments:
            message {torch.Tensor} -- A sequence to be processed, a torch.Tensor of type Long, dimensions [B, T]
        Keyword Arguments:
            lengths {Optional[torch.Tensor]} -- An optional Long tensor with messages' lengths. (default: {None})
        Returns:
            torch.Tensor -- A float tensor of [B, H]
        """
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.cell(packed)

        if isinstance(self.cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]

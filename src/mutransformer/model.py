import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from torch.utils.data import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, embedding_len, max_token_len=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_token_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_len, 2) * (-math.log(10000.0) / embedding_len))
        pe = torch.zeros(max_token_len, 1, embedding_len)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MuTransformer(nn.Module):

    def __init__(self, config):
        super(MuTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.text_prompt_ntoken = config["text_prompt_ntoken"]
        self.mubin_vocabsize = config["mubin_vocabsize"]
        self.mubin_ntoken = config["mubin_ntoken"]
        self.mubin_embedding_dim = config["mubin_embedding_dim"]

        # Trainable embedding layers for mubin inputs
        self.mubin_embedding = nn.Embedding(self.mubin_vocabsize, self.mubin_embedding_dim)

        # Positional encoding layers for mubin inputs
        self.mubin_pos_encoding = PositionalEncoding(config["pos_dropout"], self.mubin_embedding_dim, self.mubin_ntoken)

        # Transformer decoder layers
        decoder_layers = TransformerDecoderLayer(self.mubin_embedding_dim, config["nhead"], config["d_hid"], config["dropout"])
        self.transformer_decoder= TransformerDecoder(decoder_layers, config["nlayers"])

        # Final output layer
        self.output_layer = nn.Linear(self.mubin_embedding_dim, self.mubin_ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.mubin_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoded_prompt, mubin_input) -> Tensor:
        """
        Arguments:
            encoded_prompt: Tensor, shape ``[seq_len, batch_size]``
            mubin_input: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        batch_size = encoded_prompt.size(0)
        seq_len = mubin_input.size(1)
        print(f"batch_size: {batch_size}")
        print(f"seq_len: {seq_len}")
        # Embedding mubin inputs
        #print(f"mubin_input shape: {mubin_input.shape}")
        embedded_mubin = self.mubin_embedding(mubin_input)
        # Apply positional encoding to mubin embeddings
        encoded_mubin = self.mubin_pos_encoding(embedded_mubin)
        # Ensure dimensions match for concatenation
        assert encoded_prompt.size(-1) == encoded_mubin.size(-1), "Embedding last dimension does not match!"
        #print(f"encoded_prompt shape: {encoded_prompt.shape}"
        # Concatenate the distilbert encoded text and mubin inputs along the sequence length dimension
        # Generate the memory by concatenating the encoded_prompt and encoded_mubin
        memory = torch.cat((encoded_prompt, encoded_mubin), dim=1) # [batch_size, prompt_seq_len + seq_len, d_model]

        # Generate a target mask for autoregressive decoding
        tgt_seq_len = mubin_input.size(1)
        #print(f"tgt_seq_len: {tgt_seq_len}")
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(mubin_input.device)  # [seq_len, seq_len]
        print(f"tgt_mask shape: {tgt_mask.shape}")
        # Adjust dimensions to be compatible with the TransformerDecoder
        memory = memory.permute(1, 0, 2)  # [prompt_seq_len + seq_len, batch_size, d_model]
        embedded_mubin = embedded_mubin.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        # Call the transformer decoder with mubin_input as tgt and the concatenated memory
        transformer_output = self.transformer_decoder(
            tgt=embedded_mubin,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # Apply the output layer
        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        output_logits = self.output_layer(transformer_output)
        return output_logits

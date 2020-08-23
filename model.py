import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tools
from torch.nn.utils.rnn import pad_sequence
from data import TranslationDataset, SOS_token, EOS_token, MAX_LENGTH

import math

######################### PositionalEncoding ###########################

class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
        See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        assert (d_model % 2) == 0, 'd_model should be an even number.'
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

######################### ENCODER ###########################

class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(EncoderBlock, self).__init__()
        self.attn_head = nn.MultiheadAttention(n_features, n_heads,dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(n_features,n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden,n_features),
        )
        self.layer_norm2 = nn.LayerNorm(n_features)
        

    def forward(self, x, mask):
        """
        Args:
          x of shape (max_seq_length, batch_size, n_features): Input sequences.
          mask of shape (batch_size, max_seq_length): Boolean tensor indicating which elements of the input
              sequences should be ignored.
        
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        att, attn_output_weights = self.attn_head(x,x,x,mask)
        # Apply normalization and residual connection
        x = self.layer_norm1(x + self.dropout(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        # Apply normalization and residual connection
        x = self.layer_norm2(x + self.dropout(pos))
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          src_vocab_size: Number of words in the source vocabulary.
          n_blocks: Number of EncoderBlock blocks.
          n_features: Number of features to be used for word embedding and further in all layers of the encoder.
          n_heads: Number of attention heads inside the EncoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
          dropout: Dropout level used in EncoderBlock.
        """
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size,n_features)
        self.pembedding = PositionalEncoding(n_features, dropout, max_len=MAX_LENGTH)
        self.encoders = nn.ModuleList([
            EncoderBlock(n_features=n_features, n_heads=n_heads, n_hidden=n_hidden, dropout=dropout)
            for _ in range(n_blocks)])


    def forward(self, x, mask):
        """
        Args:
          x of shape (max_seq_length, batch_size): LongTensor with the input sequences.
          mask of shape (batch_size, max_seq_length): BoolTensor indicating which elements should be ignored.
        
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        x = self.embedding(x)
        x = self.pembedding(x)
        for encoder in self.encoders:
            x = encoder(x,mask)

        return x

######################### DECODER ###########################

def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class DecoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(DecoderBlock,self).__init__()
        #1
        self.masked_attn_head = nn.MultiheadAttention(n_features, n_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        
        #2
        self.attn_head = nn.MultiheadAttention(n_features, n_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(n_features)
        
        #3
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )
        self.layer_norm3 = nn.LayerNorm(n_features)
        
        #4
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, z, src_mask, tgt_mask):
        """
        Args:
          y of shape (max_tgt_seq_length, batch_size, n_features): Transformed target sequences used as the inputs
              of the block.
          z of shape (max_src_seq_length, batch_size, n_features): Encoded source sequences (outputs of the
              encoder).
          src_mask of shape (batch_size, max_src_seq_length): Boolean tensor indicating which elements of the
             source sequences should be ignored.
          tgt_mask of shape (max_tgt_seq_length, max_tgt_seq_length): Subsequent mask to ignore subsequent
             elements of the target sequences in the inputs. The rows of this matrix correspond to the output
             elements and the columns correspond to the input elements.
        
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Output tensor.

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """

        # Apply attention to inputs
        y2,_ = self.masked_attn_head(query=y,key=y,value=y,attn_mask =tgt_mask)
        y2 = self.layer_norm1(y + self.dropout(y2))
        
        # Apply attention to the encoder outputs and outputs of the previous layer
        y,_ = self.attn_head(query=y2,key=z,value=z,key_padding_mask=src_mask)# 
        y2 = self.layer_norm2(y2 + self.dropout(y))
        
        # Apply position-wise feedforward network
        y3 = self.position_wise_feed_forward(y2)
        y = self.layer_norm3(y2 + self.dropout(y3))

        return y

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          tgt_vocab_size: Number of words in the target vocabulary.
          n_blocks: Number of EncoderBlock blocks.
          n_features: Number of features to be used for word embedding and further in all layers of the decoder.
          n_heads: Number of attention heads inside the DecoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of DecoderBlock.
          dropout: Dropout level used in DecoderBlock.
        """  
        super().__init__()
        self.N = n_blocks
        self.embedding = nn.Embedding(tgt_vocab_size,n_features)
        self.pembedding = PositionalEncoding(n_features, dropout, max_len=MAX_LENGTH)
        self.decoders = nn.ModuleList([
            DecoderBlock(n_features, n_heads,
                         n_hidden, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.linear = nn.Linear(n_features,tgt_vocab_size)

         
        
    def forward(self, y, z, src_mask):
        """
        Args:
          y of shape (max_tgt_seq_length, batch_size, n_features): Transformed target sequences used as the inputs
              of the block.
          z of shape (max_src_seq_length, batch_size, n_features): Encoded source sequences (outputs of the
              encoder).
          src_mask of shape (batch_size, max_src_seq_length): Boolean tensor indicating which elements of the
             source sequences should be ignored.
        
        Returns:
          out of shape (max_seq_length, batch_size, tgt_vocab_size): Log-softmax probabilities of the words
              in the output sequences.

        Notes:
          * All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          * You need to create and use the subsequent mask in the decoder.
        """
        y = self.embedding(y)
        y = self.pembedding(y)

        for decoder in self.decoders: 
            tgt_mask = subsequent_mask(y.size(0))
            y = decoder(y=y,z=z, src_mask=src_mask,tgt_mask=tgt_mask)

        y = self.linear(y)
        y = F.log_softmax(y,dim=-1)
        return y



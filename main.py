import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformer as tr
import tools
from torch.nn.utils.rnn import pad_sequence
from data import TranslationDataset, SOS_token, EOS_token, MAX_LENGTH, collate

from model import Encoder, Decoder

n_features = 256
n_epochs = 40
batch_size = 64
skip_training=False


def training_loop(encoder,decoder,optimizer,loss_method,trainloader):
    epoch_loss = 0
    for i, data in enumerate(trainloader, 0):
        src_seqs,src_mask,tgt_seqs = data
        if src_seqs.shape[1] < batch_size:
            break    
        optimizer.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        encoder_outputs = encoder(src_seqs,src_mask.T)
        decoder_outputs= decoder(tgt_seqs[:-1], encoder_outputs, src_mask.T)
        loss = loss_method(decoder_outputs.view(-1, decoder_outputs.shape[2]),tgt_seqs[1:].view(-1))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss/len(trainloader)
             


def main():

    device = torch.device('cuda:0')

    # Create the transformer model
    encoder = Encoder(src_vocab_size=trainset.input_lang.n_words, n_blocks=3, n_features=n_features,
                    n_heads=16, n_hidden=1024)
    decoder = Decoder(tgt_vocab_size=trainset.output_lang.n_words, n_blocks=3, n_features=n_features,
                    n_heads=16, n_hidden=1024)
    encoder.to(device)
    decoder.to(device)

    # define training loop parameters
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    adam = torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = tools.NoamOptimizer(n_features, 2, 10000, adam)
    loss_method = nn.NLLLoss(ignore_index =0,reduction='mean')

    # prepare data
    data_dir = tools.select_data_dir()
    trainset = TranslationDataset(data_dir, train=True)
    trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, collate_fn=collate, pin_memory=True)

    # training
    if not skip_training:
        for epoch in range(n_epochs):
            loss = training_loop(encoder,decoder,optimizer,loss_method,trainloader)
            print(f'Train Epoch {epoch+1}: Loss: {loss}') 

    # save and load trained model
    if not skip_training:
        tools.save_model(encoder, 'tr_encoder.pth')
        tools.save_model(decoder, 'tr_decoder.pth')
    else:
        encoder = Encoder(src_vocab_size=trainset.input_lang.n_words, n_blocks=3, n_features=256, n_heads=16, n_hidden=1024)
        tools.load_model(encoder, 'tr_encoder.pth', device)
        
        decoder = Decoder(tgt_vocab_size=trainset.output_lang.n_words, n_blocks=3, n_features=256, n_heads=16, n_hidden=1024)
        tools.load_model(decoder, 'tr_decoder.pth', device) 

if __name__ == "__main__":
    main()

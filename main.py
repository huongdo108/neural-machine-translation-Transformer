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
          
             
def translate(encoder, decoder, src_seq):
    """
    Args:
      encoder (Encoder): Trained encoder.
      decoder (Decoder): Trained decoder.
      src_seq of shape (src_seq_length): LongTensor of word indices of the source sentence.
    
    Returns:
      out_seq of shape (out_seq_length, 1): LongTensor of word indices of the output sentence.
    """
    encoder.eval()
    decoder.eval()
    # src_seq = torch.tensor(src_seq).unsqueeze(dim=1)
    src_seq= src_seq.clone().detach()
    src_mask = (src_seq == 0).T
    e_output = encoder(src_seq, src_mask)
    out_seq = torch.zeros(MAX_LENGTH).unsqueeze(dim=1).type(torch.LongTensor)
    j = 0
    for i in range(MAX_LENGTH - 1):
        decoder_out = decoder(out_seq[0 : i + 1], e_output, src_mask)
        index = [torch.argmax(h).item() for h in decoder_out]
        out_seq[i + 1] = index[i]
        if index[i] == 1:
            j = i + 1
            break

    return out_seq[1 : j + 1]


class NoamOptimizer:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()


def main():

    device = torch.device('cuda:0')
    n_features = 256
    n_epochs = 40
    batch_size = 64
    skip_training=False

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
    optimizer = NoamOptimizer(n_features, 2, 10000, adam)
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
        tools.save_model(encoder, 'tr_encoder.pth')
        tools.save_model(decoder, 'tr_decoder.pth')
    else:
        encoder = Encoder(src_vocab_size=trainset.input_lang.n_words, n_blocks=3, n_features=256, n_heads=16, n_hidden=1024)
        tools.load_model(encoder, 'tr_encoder.pth', device)
        
        decoder = Decoder(tgt_vocab_size=trainset.output_lang.n_words, n_blocks=3, n_features=256, n_heads=16, n_hidden=1024)
        tools.load_model(decoder, 'tr_decoder.pth', device) 

    # Generate translations with the trained model

    # translate sentences from the training set
    print('Translate training data:')
    print('-----------------------------')
    for i in range(5):
        src_sentence, tgt_sentence = trainset[np.random.choice(len(trainset))]
        print('>', ' '.join(trainset.input_lang.index2word[i.item()] for i in src_sentence))
        print('=', ' '.join(trainset.output_lang.index2word[i.item()] for i in tgt_sentence))
        out_sentence = translate(encoder, decoder, src_sentence)
        print('<', ' '.join(trainset.output_lang.index2word[i.item()] for i in out_sentence), '\n')

    # translate sentences from the test set
    testset = TranslationDataset(data_dir, train=False)
    print('Translate test data:')
    print('-----------------------------')
    for i in range(5):
        input_sentence, target_sentence = testset[np.random.choice(len(testset))]
        print('>', ' '.join(testset.input_lang.index2word[i.item()] for i in input_sentence))
        print('=', ' '.join(testset.output_lang.index2word[i.item()] for i in target_sentence))
        output_sentence = translate(encoder, decoder, input_sentence)
        print('<', ' '.join(testset.output_lang.index2word[i.item()] for i in output_sentence), '\n')

if __name__ == "__main__":
    main()

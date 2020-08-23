import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn
from data import MAX_LENGTH


def select_data_dir(data_dir="../data"):
    data_dir = "/coursedata" if os.path.isdir("/coursedata") else data_dir
    print("The data directory is %s" % data_dir)
    return data_dir


def save_model(model, filename):
    try:
        do_save = input("Do you want to save the model (type yes to confirm)? ").lower()
        if do_save == "yes":
            torch.save(model.state_dict(), filename)
            print("Model saved to %s." % (filename))
        else:
            print("Model not saved.")
    except:
        raise Exception("The notebook should be run or validated with skip_training=True.")


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print("Model loaded from %s." % filename)
    model.to(device)
    model.eval()


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
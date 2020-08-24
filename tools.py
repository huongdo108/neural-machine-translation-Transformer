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



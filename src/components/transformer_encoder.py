import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import pdb
from src.components.utils import *


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn"

    def __init__(self, self_attn):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        #self.feed_forward = feed_forward

    def forward(self, x, mask):
        return self.self_attn(x, x, x, mask)

class EncoderLayerFFN(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, self_attn, feed_forward):
        super(EncoderLayerFFN, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

    def forward(self, x, mask):
        return self.feed_forward(self.self_attn(x, x, x, mask))

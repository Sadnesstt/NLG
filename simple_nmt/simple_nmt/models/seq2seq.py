import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchBoard

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
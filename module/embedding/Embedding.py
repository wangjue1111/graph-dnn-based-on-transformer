import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size,pading_idx):
        super().__init__(vocab_size, embed_size, padding_idx=pading_idx)
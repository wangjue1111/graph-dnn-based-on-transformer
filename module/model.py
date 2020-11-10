from .encoder import Encoder
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,input_size,embedding_size,num_layers,num_heads,hidden,max_len,target_size,dropout):
        super(Model, self).__init__()
        self.embedding=nn.Linear(input_size,embedding_size)
        self.encoder=Encoder(embedding_size,num_layers,num_heads,max_len,hidden,dropout)
        self.output=nn.Sequential(nn.Linear(embedding_size,target_size),nn.LogSoftmax())

    def forward(self,x,mask=None,dropout=None):
        x=self.embedding(x)
        output=self.encoder(x,mask,dropout)
        return self.output(output)

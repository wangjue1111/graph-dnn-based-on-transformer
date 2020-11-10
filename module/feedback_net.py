import torch
import torch.nn as nn

class Feedback_Net(nn.Module):
    def __init__(self,embedding_size,hidden,dropout):
        super(Feedback_Net, self).__init__()
        self.norm=nn.LayerNorm(embedding_size)
        self.layer=nn.Sequential(nn.Linear(embedding_size,hidden),nn.GELU())
        self.output=nn.Sequential(nn.Linear(hidden,embedding_size),nn.GELU())
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,dropout):
        x=self.norm(x)
        output=self.layer(x)
        if dropout:
            return self.dropout(self.output(output))+x
        return self.output(output)+x
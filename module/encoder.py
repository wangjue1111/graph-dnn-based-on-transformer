import torch.nn as nn
from .attention import Multihead_Attention
from .feedback_net import Feedback_Net
from checkpoint.cora.embedding import PositionalEmbedding

class Encoder_Layer(nn.Module):
    def __init__(self,embedding_size,num_heads,max_len,hidden,dropout):
        super(Encoder_Layer, self).__init__()
        self.attn=Multihead_Attention(embedding_size,num_heads,max_len,dropout)
        self.net=Feedback_Net(embedding_size,hidden,dropout)

    def forward(self,h,h_mask,dropout=None):
        h=self.attn(h,h,h,h_mask,dropout)
        h=self.net(h,dropout)
        return h

'''
x,g,pos,seg=None,h_mask=None,g_mask=None
'''

class Encoder(nn.Module):
    def __init__(self,embedding_size,num_layers,num_heads,max_len,hidden,dropout):
        super(Encoder, self).__init__()
        self.max_len=max_len
        self.embedding_size=embedding_size
        self.positional_embedding=PositionalEmbedding(embedding_size,max_len)
        self.encoder_layers=nn.ModuleList([Encoder_Layer(embedding_size,num_heads,max_len,hidden,dropout) for i in range(num_layers)])
#        self.output=nn.Linear(embedding_size,output_size)

#        self.to_vocab=nn.Sequential(nn.Linear(output_size,token_vocab_size),nn.Softmax())

    def forward(self,h,h_mask,dropout=None):

        for encoder_layer in self.encoder_layers:
            h=encoder_layer(h,h_mask,dropout)
#        outputs=self.output(h)
        return h
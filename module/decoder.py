import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Multihead_Attention
from .feedback_net import Feedback_Net
from .embedding.Embedding import Embedding
import random
from .embedding.Positional_Embedding import PositionalEmbedding

class Decoder_Layer(nn.Module):
    def __init__(self,embedding_size,num_heads,max_len,hidden,dropout,prob=0.5):
        super(Decoder_Layer, self).__init__()
        self.attn=Multihead_Attention(embedding_size,num_heads,max_len,dropout)
        self.net=Feedback_Net(embedding_size,hidden,dropout)
        self.prob=prob

    def forward(self,h,encoder_outputs,h_mask,src_mask,dropout=None):
        if random.random()<self.prob:
            h=self.attn(h,h,h,h_mask,dropout)

        h=self.attn(h,encoder_outputs,encoder_outputs,src_mask,dropout)

        h=self.net(h,dropout)
        return h

'''
x,g,pos,seg=None,h_mask=None,g_mask=None
'''

class Decoder(nn.Module):
    def __init__(self,embedding_size,num_layers,num_heads,max_len,hidden,y1_vocab_size,y2_vocab_size,domain_vocab_size,query_type_vocab_size,padding_index,dropout,prob=0.5):
        super(Decoder, self).__init__()
        self.max_len=max_len
        self.embedding_size=embedding_size
#        self.seg_embedding=Embedding(seg_vocab_size,embedding_size,padding_index)
        self.y1_embedding=Embedding(y1_vocab_size,embedding_size,padding_index)
        self.y2_embedding=Embedding(y2_vocab_size,embedding_size,padding_index)
        self.query_type_embedding=Embedding(query_type_vocab_size,embedding_size,padding_index)
        self.is_mul_embedding=Embedding(5,embedding_size,padding_index)
        self.domain_embedding=Embedding(domain_vocab_size,embedding_size,padding_index)
        self.positional_embedding=PositionalEmbedding(embedding_size,max_len)
        self.decoder_layers=nn.ModuleList([Decoder_Layer(embedding_size,num_heads,max_len,hidden,dropout,prob=prob) for i in range(num_layers)])
#        self.output=nn.Linear(embedding_size,output_size)
      #  self.token_vocab_size=token_vocab_size
#        self.to_vocab=nn.Sequential(nn.Linear(output_size,token_vocab_size),nn.Softmax())
        self.init_embedding=nn.Parameter(torch.normal(0.0,0.5,size=[1,1,embedding_size]))

    def forward(self,h,encoder_outputs,attn_mask,src_mask,name,dropout=None):
        if name=='y1':
            h=self.y1_embedding(h)
        elif name=='y2':
            h=self.y2_embedding(h)
        else:
            h=self.init_embedding.repeat([h.shape[0],1,1])

        pos_emb=self.positional_embedding(h)
        h=h+pos_emb
        for decoder_layer in self.decoder_layers:
            h=decoder_layer(h,encoder_outputs,attn_mask,src_mask,dropout)
#        outputs=self.output(h)
        return h
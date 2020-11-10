import torch
import torch.nn.functional as F
import torch.nn as nn

def attention(q,k_h,v,mask=None,dropout=None):
    attn=torch.matmul(q,k_h.transpose(-1,-2))

    if mask!=None:
        attn-=mask*1e30
    attn_score=F.softmax(attn,-1)

    if dropout:
        attn_score=dropout(attn_score)

    return torch.matmul(attn_score,v)

class Multihead_Attention(nn.Module):
    def __init__(self,embedding_size,num_heads,length,dropout):
        super(Multihead_Attention, self).__init__()
        self.embedding_size=embedding_size
        self.num_heads=num_heads
        self.length=length
        self.d=embedding_size//num_heads
        self.q_layer=nn.Linear(embedding_size,embedding_size)
        self.k_layer=nn.Linear(embedding_size,embedding_size)
        self.v_layer=nn.Linear(embedding_size,embedding_size)
        self.dropout=nn.Dropout(dropout)
        self.output=nn.Linear(embedding_size,embedding_size)
        self.norm=nn.LayerNorm(embedding_size)

    def change2head(self,value,name,dropout=None):
        if name=='q':
            layer=self.q_layer
        elif name=='k':
            layer=self.k_layer
        elif name=='v':
            layer=self.v_layer
        value=layer(value)
        value=value.view(self.batch_size,-1,self.num_heads,self.d).transpose(1,2)

        return value

    def forward(self,q,k,v,h_mask=None,dropout=None):
        self.batch_size=q.shape[0]
        q,k,v=self.norm(q),self.norm(k),self.norm(v)
        x_q=self.change2head(q,'q')
        x_k=self.change2head(k,'k')
        x_v=self.change2head(v,'v')

        if dropout:
            h_value=attention(x_q,x_k,x_v,h_mask,self.dropout)
        else:
            h_value=attention(x_q,x_k,x_v,h_mask)
        h_value=h_value.transpose(1,2).contiguous().view(self.batch_size,-1,self.embedding_size)

        if dropout:
            return self.dropout(self.output(h_value))+q
        return self.output(h_value)+q

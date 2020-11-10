import math
import numpy as np
import torch
class Vocab(object):
    def __init__(self,device=None):
        self.s2i={'_PAD':0,'_UNK':1,'<s>':2}
        self.i2s=['_PAD','_UNK','<s>']
        self.p_c=None
        self.device=device
        self.special_label=['_PAD','_UNK','<s>']
        self.c_num=[]
        self.if_p_c=False
    def s2i_f(self,words):
        for word in words:
            if word not in self.s2i and word:
                self.s2i[word]=len(self.i2s)
                self.i2s.append(word)

            index=self.s2i.get(word,1)


    def i2s_f(self,seq):
        res=''
        if len(seq.shape)==0:
            seq=[seq]
        for index in seq:
            res+=self.i2s[int(index)]+' '
        return res

    def s2i2i2s(self):
        self.i2s=[]
        for k in self.s2i:
            self.i2s.append(k)

    def get_p(self):
        self.p_c=torch.ones([len(self.s2i,)]).to(self.device)*(math.e-1)
        self.c_num=torch.zeros([len(self.s2i,)]).to(self.device)
'''
        if not self.if_p_c:
            sum_=sum(self.p_c)
            for i in range(3,len(self.p_c)):
                tmp=self.p_c[i]/sum_
                tmp=1/math.log(1.02+tmp,math.exp(1))
                self.p_c[i]=max(min(tmp,50.0),1.0)
        self.if_p_c = True
'''

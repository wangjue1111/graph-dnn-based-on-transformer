from collections import defaultdict
from vocab import Vocab
import numpy as np
import torch
import yaml
from module.model import Model
from sklearn.metrics import f1_score
from utils import *

data1=open('checkpoint/Pubmed-Diabetes/new_data/Pubmed-Diabetes.DIRECTED.cites.tab').readlines()
datas=open('checkpoint/Pubmed-Diabetes/new_data/Pubmed-Diabetes.NODE.paper.tab').readlines()

config=yaml.load(open('config.yaml').read())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

map4index={}

graph=defaultdict(list)
x_vocab=Vocab()
y_vocab=Vocab()
seq_size=config['seq_size']
length=len(datas)
batch_num=config['batch']
dropout=config['dropout']
input_size=config['input_size']
embedding_size=config['embedding_size']
num_layers=config['num_layers']
num_heads=config['num_heads']
hidden=config['hidden']
epoch=config['epoch']
devices=[i for i in range(config['devices'])]

for data in data1:
    data=data.strip().split('\t')
    graph[data[0]].append(data[1])
    graph[data[1]].append(data[0])
    x_vocab.s2i_f(data)

for i,data in enumerate(datas):
    x=data.split('\t')[0]
    map4index[x]=i
    data=data.split(('\t'))[-1]
    y_vocab.s2i_f([data])

def create_seq(items,vocab,max_len,flag=0):
    res=[]
    if flag==1:
        items=['<s>']+items
    elif flag==2:
        items=items+['</s>']

    for item in items:

        res.append(vocab.get(item,vocab['_UNK']))
    length=len(res)
    if len(res)<max_len:
        res+=[vocab['_PAD'] for i in range(max_len-len(res))]
    elif len(res)>max_len:
        res=res[:max_len]
    return res

def generate_seq(datas,all_datas=None,flag=False):
    indexes=np.random.choice(list(range(len(datas))),seq_size,replace=False)
    xs=[datas[index].split('\t')[0] for index in indexes]
    ys=[datas[index].split('\t')[-1] for index in indexes]
    xss=[list(map(float,datas[index].split('\t')[1:-1])) for index in indexes]
    mask=1-np.eye(seq_size)[np.newaxis,:]
    for i,index in enumerate(indexes):
        x=datas[index].split('\t')[0]
        for item in graph[x]:
            if item in xs:
                ind=xs.index(item)
                mask[0][i][ind]=0
                mask[0][ind][i]=0
    ys=create_seq(ys,y_vocab.s2i,seq_size)
    return xss,ys,torch.from_numpy(mask),indexes

model=Model(input_size,embedding_size,num_layers,num_heads,hidden,seq_size,len(y_vocab.s2i),dropout).to(device)
loss_fn=torch.nn.NLLLoss().to(device)
optim=torch.optim.Adam(model.parameters(),lr=0.0002)

if torch.cuda.is_available():
    model=torch.nn.DataParallel(model,device_ids=devices)
test_datas=datas[:200]
train_datas=datas[200:]

for i in range(epoch):
    xs,ys,masks=[],[],[]
    for j in range(batch_num):
        x,y,mask,_=generate_seq(train_datas)
        xs.append(x)
        ys.append(y)
        masks.append(mask.unsqueeze(0))
    xs=torch.FloatTensor(xs).to(device)
    ys=torch.LongTensor(ys).to(device)
    masks=torch.cat(masks).to(device)
    predict=model(xs.to(device),masks.to(device),dropout)
    loss=loss_fn(predict.view(-1,len(y_vocab.s2i)),ys.view(-1))
    print(loss)
    print(predict.argmax(-1).view(-1)[0])
    print(ys.view(-1)[0])
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i%1==0:
        xs, ys, masks = [], [], []
        all_indexes=[]
        for j in range(batch_num):
            x, y, mask,indexes = generate_seq(datas)
            xs.append(x)
            ys.append(y)
            masks.append(mask.unsqueeze(0))
            all_indexes.append(indexes<200)
        xs = torch.FloatTensor(xs)
        ys = torch.LongTensor(ys)
        masks = torch.cat(masks)
        predict = model(xs.to(device))
        all_indexes=np.concatenate(all_indexes,0)
        ys=ys.view(-1).cpu()[all_indexes]
        print('f1_score:',f1_score(ys, predict.argmax(-1).view(-1).cpu()[all_indexes], average='micro'))

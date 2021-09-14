#!/usr/bin/env python
# coding: utf-8

# # Map Heuristic Approximation
# 
# The goal is to use neural networks to approximate map heuristics
# 
# Let's start simple. Treat the problem as a regression problem for each cell and use a simple fully connected neural network. Let's see what this network can do.

# In[30]:


import torch
import torch.nn as nn
import numpy as np

class MapDataLoader(torch.utils.data.Dataset):
    def __init__(self,file,root="./"):
        self.data = np.load(os.path.join(root,file))
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem_(self,index):
        return self.data[index][0],self.data[index][1]


# In[97]:


from functools import reduce

class FullyConnectedNetwork(nn.Module):
    def __init__(self,inputs,outputs,nodesPerLayer):
        super(FullyConnectedNetwork,self).__init__()
        self.layers = []
        i = inputs
        for j in nodesPerLayer:
            self.layers.append(nn.Linear(i,j))
            self.layers.append(nn.ReLU())
            i = j
        self.layers.append(nn.Linear(i,outputs))
        print(self.layers)
        
    def forward(self,x):
        y = self.layers[0](x)
        y = self.layers[1](y)
        y = self.layers[2](y)
        y = self.layers[3](y)
        return y
        #return reduce(lambda y,l:l(y),self.layers,x)   
    
class Network(nn.Module):
    def __init__(self,inputs,outputs,hidden):
        super(Network,self).__init__()
        self.layers=[]
        self.layers.append(nn.Linear(inputs,hidden))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden,outputs))
        self.fc1 = nn.Linear(inputs,hidden)
        print(self.layers)
        #print(self.fc1)
        
    def forward(self,x):
        y = self.layers[0](x)
        y = self.layers[1](y)
        y = self.layers[2](y)
        return y
        


# In[103]:


#network = FullyConnectedNetwork(100,100,[150,100])
network = Network(100,100,200)
network.parameters()




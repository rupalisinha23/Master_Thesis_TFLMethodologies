import sys
import torch
import torch.nn as nn
from box import Box
from einops import rearrange, reduce
from torch.autograd import Variable
from torch.nn import functional as F

# model

class exp_tagger(nn.Module):
    def __init__(self,flag,embeddings_from_ft, params, outputlayers, mtl_type):
        super(exp_tagger, self).__init__()
        self.conf = Box(params, default_box=True, default_box_attr=None)
        self.word_embedding = nn.Embedding.from_pretrained(embeddings_from_ft)
#         self.word_embedding = nn.Embedding(self.conf.vocab_size[0],self.conf.embedding_dims[0], padding_idx=0)
        self.relu = nn.ReLU()
        self.outputlayers = outputlayers
        self.mtl_type = mtl_type
        self.flag = flag
        
        
        if type(self.outputlayers) != list:
            self.label_size1 = self.conf.values_size[0] if self.outputlayers == 'values' else self.conf.relations_size[0] if self.outputlayers == 'relations' else self.conf.concepts_size[0]
            
            if mtl_type == 'dense-A':
                self.label_embedding1 = nn.Embedding(self.label_size1, self.conf.embedding_dims[0], padding_idx=0)
                
            
        elif len(self.outputlayers) == 3:
            self.label_size1 = self.conf.values_size[0] if self.outputlayers[0] == 'values' else self.conf.relations_size[0] if self.outputlayers[0] == 'relations' else self.conf.concepts_size[0]
            self.label_size2 = self.conf.values_size[0] if self.outputlayers[1] == 'values' else self.conf.relations_size[0] if self.outputlayers[1] == 'relations' else self.conf.concepts_size[0]
            self.label_size3 = self.conf.values_size[0] if self.outputlayers[2] == 'values' else self.conf.relations_size[0] if self.outputlayers[2] == 'relations' else self.conf.concepts_size[0]
            
        
        # layers
        self.FC1 = nn.Linear(self.conf.embedding_dims[0], 100)
        self.FC2 = nn.Linear(100,1400)
        self.FC3 = nn.Linear(1400,50)
                                                

        
        if self.flag == "STL":
            self.FC_out = nn.Linear(50, self.label_size1)
        elif self.flag == "MTL":
            if self.mtl_type is None:
                print('Input the MTL type out of A/B/C/D/E!')
                sys.exit()
            else:
                
                if self.mtl_type == 'A':
                    self.FC_out1 = nn.Linear(50,self.label_size1)
                    self.FC_out2 = nn.Linear(50, self.label_size2)
                    self.FC_out3 = nn.Linear(50, self.label_size3)
                

    def forward(self, batch_sentences):
        
        embeds = self.word_embedding(batch_sentences)
        tmp = F.avg_pool2d(embeds, (embeds.shape[1], 1)).squeeze(1)
        
        
        if self.flag == "STL":
            label_score = self.relu(self.FC3(self.relu(self.FC2(self.relu(self.FC1(tmp))))))
            binary_score1 = self.FC_out(label_score)
            return binary_score1
        
        elif self.flag == "MTL":
            if self.mtl_type == "A":
                label_score = self.relu(self.FC3(self.relu(self.FC2(self.relu(self.FC1(tmp))))))
                binary_score1 = self.FC_out1(label_score)
                binary_score2 = self.FC_out2(label_score)
                binary_score3 = self.FC_out3(label_score)
                return binary_score1, binary_score2, binary_score3
 


class CNN(nn.Module):
#     flag,weights,cnn_params, params
    def __init__(self, unique_labels, embeds_from_ft,cnn_params, params): 
        super(CNN, self).__init__()
        self.conf = Box({**cnn_params, **params}, default_box=True, default_box_attr=None)
    
        self.relu = nn.ReLU()
    
        self.embedding = nn.Embedding.from_pretrained(embeds_from_ft)
       
        self.convs = nn.ModuleList([nn.Conv1d(self.conf.emb_dim, self.conf.num_filters, kernel_size=size) for size in self.conf.filter_sizes])
        
        if len(unique_labels) == 1:
            self.label_size1 = self.conf.values_size[0] if unique_labels[0] == 'values' else self.conf.relations_size[0] if unique_labels[0] == 'relations' else self.conf.concepts_size[0]
        else:
            self.label_size1 = self.conf.values_size[0] if unique_labels[0] == 'values' else self.conf.relations_size[0] if unique_labels[0] == 'relations' else self.conf.concepts_size[0]
            self.label_size2 = self.conf.values_size[0] if unique_labels[1] == 'values' else self.conf.relations_size[0] if unique_labels[1] == 'relations' else self.conf.concepts_size[0]
            self.label_size3 = self.conf.values_size[0] if unique_labels[2] == 'values' else self.conf.relations_size[0] if unique_labels[2] == 'relations' else self.conf.concepts_size[0]

        
        
        self.FC1 = nn.Linear(len(self.conf.filter_sizes) * self.conf.num_filters, 100)
        self.FC2 = nn.Linear(100,1400)
        self.FC3 = nn.Linear(1400,50)
        
        if len(unique_labels) == 1:
            self.FC_out1 = nn.Linear(50, self.label_size1)
        else:
            self.FC_out1 = nn.Linear(50, self.label_size1)
            self.FC_out2 = nn.Linear(50, self.label_size2)
            self.FC_out3 = nn.Linear(50, self.label_size3)
        if self.conf.dropout is not None:
            self.dropout = nn.Dropout(self.conf.dropout)
            
    def forward(self,x):
        x = rearrange(x, 'batch seqlen -> batch seqlen')
        emb = rearrange(self.embedding(x), 'batch seqlen embdim -> batch embdim seqlen')
        pooled = [reduce(conv(emb), 'batch embdim seqlen -> batch embdim', 'max') for conv in self.convs]    # max pool over n-gram convolutions
        concatenated = rearrange(pooled, 'numfiltersizes batch nfilters -> batch (numfiltersizes nfilters)') # concatenate filters per filter size-- e.g. if batch=10 and filter_sizes(1,2) => 2 filter sizes and n_filters=4, then (2, 10, 4) -> (10, 2x4=8) 
        act = torch.relu(concatenated) # delayed relu that works because argmax(linear) = argmax(relu(linear)) give the same neurons
        if self.conf.dropout: # only activates if (not None) and (not 0)
            act = self.dropout(act)
        
        label_score = self.relu(self.FC1(act))
        label_score = self.relu(self.FC2(label_score))
        label_score = self.FC3(label_score)
          
        if self.conf.flag =='STL':
            binary_score1 = self.FC_out1(label_score)
            return binary_score1
        elif self.conf.flag == 'MTL':
            binary_score1 = self.FC_out1(label_score)
            binary_score2 = self.FC_out2(label_score)
            binary_score3 = self.FC_out3(label_score)
           
            return binary_score1, binary_score2, binary_score3
            
           
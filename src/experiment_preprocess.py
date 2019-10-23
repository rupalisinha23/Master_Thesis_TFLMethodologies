import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import torch
import pickle
import codecs
import sys
from textblob_de import TextBlobDE
from sklearn.preprocessing import MultiLabelBinarizer


class preprocess(object):
    
    
    @staticmethod
    def create_sen_indices(word2idx, data):
        """
        function to convert a sentence to a list of indices
        :param word2idx: word to index mapping dictionary
        :param data: data(which needs to be transformed:list of sentences)
        :return list of list of sentences where each word is replaces by its index values from word2idx.
        """
        idx = []
        sen_indices = []
        for sen in data:
            blob = TextBlobDE(sen)

            for w in blob.tokens:
                if w not in word2idx:
                    idx.append(word2idx['<UNK>'])
                else:
                    idx.append(word2idx[w])
            sen_indices.append(idx)
            idx = []
        return sen_indices
    
    
    @staticmethod
    def convert2type(x_train, x_val, x_test, y_train,y_val, y_test):
        """
        function to convert x data to torch.LongTensor and y data to torch.FloatTensor
        :param x_train, x_val, x_test: x data
        :param y_train, y_val, y_test: y data
        :return corresponding x and y data after converting each of them to a suitable type
        """
         
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
                
        x_train = torch.LongTensor(x_train).to(device)
        x_val = torch.LongTensor(x_val).to(device)
        x_test = torch.LongTensor(x_test).to(device)

        y_train_concepts = torch.FloatTensor(y_train['concepts_onehot'].values.tolist()).to(device)
        y_train_values = torch.FloatTensor(y_train['values_onehot'].values.tolist()).to(device)
        y_train_relations = torch.FloatTensor(y_train['relations_onehot'].values.tolist()).to(device)

        y_val_concepts = torch.FloatTensor(y_val['concepts_onehot'].values.tolist()).to(device)
        y_val_values = torch.FloatTensor(y_val['values_onehot'].values.tolist()).to(device)
        y_val_relations = torch.FloatTensor(y_val['relations_onehot'].values.tolist()).to(device)

        y_test_concepts = torch.FloatTensor(y_test['concepts_onehot'].values.tolist()).to(device)
        y_test_values = torch.FloatTensor(y_test['values_onehot'].values.tolist()).to(device)
        y_test_relations = torch.FloatTensor(y_test['relations_onehot'].values.tolist()).to(device)

        return x_train, x_val, x_test, y_train_concepts, y_train_values, y_train_relations, y_val_concepts, y_val_values, y_val_relations, y_test_concepts, y_test_values, y_test_relations
    
    
    @staticmethod
    def create_class_names(df, path_to_classes):
        """
        read a pickle file which holds the class names of each of the y data
        :return target names of concepts and relations
        """
#         with open('/home/rusi02/research_paper/big_medilytics/data/classes_v1.pkl', 'rb') as f:
#             combined_list = pickle.load(f)
        with open(path_to_classes, 'rb') as f:
            combined_list = pickle.load(f)

        return combined_list[0], combined_list[1], combined_list[2]
    
    
    def read_vecfile_for_embeddings(self,path):
        """
        method to read the .vec file and create a dictionary with word as key and a list of index and
        embedding as value pair.   
        :param path: path to the .vec file
        :return: word2idx: a dictionary with word as key and index as value
        embedding_matrix: a list of ist consisting of all the embeddings of the words
        word2idx_embeds: a dictionary where key is the word in vocabulary and the value is a list of index and embedding
        """

        count = 2
        embedding_dims = 100
        embedding_matrix=[]
        word2idx_embeds = {'<PAD>':[0, [float(0)]*embedding_dims],'<UNK>':[1]}
        with codecs.open(path, 'rb', encoding="utf-8", errors="ignore") as vecs:
            next(vecs)  # skip count header
            for vec in vecs:
                data = vec.strip().split(' ')
                # extract embeddings
                
                if data[0] == '<PAD>':
                    word2idx_embeds['<PAD>'] = word2idx['<PAD>'].append(float(0)*embedding_dims)
                elif data[0] == '<UNK>':
                    word2idx_embeds['<UNK>'] = [1,[float(i) for i in data[1:]]]
                else:
                    word2idx_embeds[data[0]] = [count, [float(i) for i in data[1:]]]
                    count +=1
                
        embedding_matrix = [word2idx_embeds['<PAD>'][1], word2idx_embeds['<UNK>'][1]]
        word2idx = {'<PAD>':0, '<UNK>':1}
        for key, item in word2idx_embeds.items():
            if key == '<PAD>':
                pass
            elif key == '<UNK>':
                pass
            else:
                word2idx[key] = item[0]
                embedding_matrix.append(item[1])
        return word2idx, embedding_matrix, word2idx_embeds
    
    
    def create_data(self,path, word2idx, path_to_classes):
        """
        method to split the data into training, validation and test sets. Then create the sentence as a list of word indices, pad them and call the ype conversion onto them. Also include the target label names.
        :param path: path to the pickle file which stores the dataframe
        :param word2idx: word to index dictionary
        :return: corresponding splits into the required format
        """
        
        df = pd.read_pickle(path)
        
 
        x, x_test, y, y_test = train_test_split(df['sentence'], df[['concepts','relations',
                                                                    'concepts_onehot', 'values_onehot',
                                                                    'relations_onehot']], test_size=0.10,
                                                                    train_size=0.90,
                                                                    random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.11, train_size=0.89, random_state=42)
        
        
        
        # transform sentences as list of word indices
        x_train = preprocess.create_sen_indices(word2idx,x_train)
        x_val = preprocess.create_sen_indices(word2idx,x_val)
        x_test = preprocess.create_sen_indices(word2idx,x_test)
        
        # pad input data
        max_features_train = max([max(i) for i in x_train])
        max_features_val = max([max(i) for i in x_val])
        max_features_test = max([max(i) for i in x_test])

        x_train = sequence.pad_sequences(x_train, maxlen=max_features_train, padding='post')
        x_val = sequence.pad_sequences(x_val, maxlen=max_features_val, padding='post')
        x_test = sequence.pad_sequences(x_test, maxlen=max_features_test, padding='post')

        # convert x--> longtensor and y --> floattensor; push to gpu
        x_train, x_val, x_test, y_train_concepts, y_train_values, y_train_relations, y_val_concepts, y_val_values, y_val_relations, y_test_concepts, y_test_values, y_test_relations = preprocess.convert2type(x_train, x_val, x_test, y_train, y_val, y_test)
        
        # create onehot class names
        concept_class, values_class, relation_class = preprocess.create_class_names(df, path_to_classes)
        
        return {'x_train': x_train,
               'x_val': x_val,
               'x_test': x_test,
               'y_train_concepts': y_train_concepts,
               'y_train_values': y_train_values,
               'y_train_relations': y_train_relations,
               'y_val_concepts': y_val_concepts,
               'y_val_values': y_val_values,
               'y_val_relations': y_val_relations,
               'y_test_concepts': y_test_concepts,
               'y_test_values': y_test_values,
               'y_test_relations': y_test_relations,
               'concepts_class': concept_class,
               'values_class': values_class,
               'relations_class': relation_class}
    
    
    def get_length(self, y_concepts,y_relations):
        """
        method to calculate the length of the target labels: concepts and relations
        """
        return len(y_concepts[0]), len(y_relations[0])
    

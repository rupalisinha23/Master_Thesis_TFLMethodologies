import create_df as cd
import pandas as pd
import sys
import copy
from nltk import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from textblob_de import TextBlobDE as TextBlob
from itertools import zip_longest
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


# accessing the df variable from create_df.py
df = cd.df
mlb = MultiLabelBinarizer()

def filter_concepts(onehot_concepts, onehot_concepts_classes) -> list:
    """
    Filters the infrequent concepts which are less than 500.
    Also filters the 'Negation' concept as suggested by Roland.
    """
    
    onehot = [sum(x) for x in zip(*onehot_concepts)]
    list_above_10 = []
    for i, val in enumerate(onehot):
        if val > 500:
            list_above_10.append(i)
    t = []
    temporary = []
    for outer in onehot_concepts:
        for idx,val in enumerate(outer):
            if idx in list_above_10:
                temporary.append(val)
        t.append(temporary)
        temporary=[]
    onehot_concepts = t

    x=[]
    for idx, val in enumerate(onehot_concepts_classes):
        if idx in list_above_10:
            x.append(val)
    onehot_concepts_classes = x

    
    for row in onehot_concepts:
        del row[onehot_concepts_classes.index('Negation')]
    
    onehot_concepts_classes.remove('Negation')
    return onehot_concepts, onehot_concepts_classes
            

def create_onehot_concepts(mlb,label):
    
    onehot_concepts = mlb.fit_transform([set(i) for i in df[label]])
    onehot_concepts_classes = mlb.classes_
  
#     onehot_concepts, onehot_concepts_classes = filter_concepts(onehot_concepts, onehot_concepts_classes)

    kom_idx, tmp_idx = onehot_concepts_classes.tolist().index('Kommentar'), onehot_concepts_classes.tolist().index('Temporal_course')

    onehot_one = onehot_concepts[:,0:kom_idx]
    onehot_two = onehot_concepts[:,kom_idx+1:tmp_idx]
    onehot_last = onehot_concepts[:, tmp_idx+1:]
    onehot_concepts = np.concatenate((onehot_one, onehot_two), axis=1)
    onehot_concepts = np.concatenate((onehot_concepts, onehot_last), axis=1)
            
    l = []
    for idx, val in enumerate(onehot_concepts_classes):
        if idx == kom_idx:
            pass
        elif idx == tmp_idx:
            pass
        else:
            l.append(val)
    onehot_concepts_classes = l

    return onehot_concepts, onehot_concepts_classes
    




def create_onehot_values(mlb, label):
    # one hot encoding for values only
    a_tmp = []
    a_list = []
    for i in df[label]:
        for j in i:
            a_tmp.append(j[1])
        a_list.append(a_tmp)
        a_tmp=[]
    onehot_values = mlb.fit_transform([set(i) for i in a_list])
    onehot_values_classes = mlb.classes_
#     onehot_values = onehot_values.tolist()
#     onehot_values_classes = onehot_values_classes.tolist()
    
#     name = ['ICD10','MDR','cD','cE','sD']
#     idx = []
    
#     [idx.append(onehot_values_classes.index(n)) for n in name]
    
#     onehot_outer = []
    
#     for item in onehot_values:
#         onehot_inner = []
#         for ind, value in enumerate(item):
#             if ind not in idx:
#                 onehot_inner.append(value)
#         onehot_outer.append(onehot_inner)
#     onehot_values = onehot_outer
            
#     onehot_values_classes.remove('ICD10')
#     onehot_values_classes.remove('MDR')
#     onehot_values_classes.remove('cD')
#     onehot_values_classes.remove('cE')
#     onehot_values_classes.remove('sD')
    
    return onehot_values, onehot_values_classes
    



def create_onehot_relations(mlb, label):
    a_tmp = []
    a_list = []
    for i in df['relations']:
        for j in i:
            a_tmp.append(j[0])
        a_list.append(a_tmp)
        a_tmp = []
    onehot_relations = mlb.fit_transform(set(i) for i in a_list)
    onehot_relations_classes = mlb.classes_
#     print(onehot_relations_classes, len(onehot_relations_classes))
#     onehot_relations = onehot_relations.tolist()
#     onehot_relations_classes = onehot_relations_classes.tolist()

#     name = ['hasState','has_dosing','has_measure','has_time_info','is_located','is_specified']
#     idx = []
    
#     [idx.append(onehot_relations_classes.index(n)) for n in name]
    
#     onehot_outer = []
    
#     for item in onehot_relations:
#         onehot_inner = []
#         for ind, value in enumerate(item):
#             if ind not in idx:
#                 onehot_inner.append(value)
#         onehot_outer.append(onehot_inner)
        
#     onehot_relations = onehot_outer
            
#     onehot_relations_classes.remove('hasState')
#     onehot_relations_classes.remove('has_dosing')
#     onehot_relations_classes.remove('has_measure')
#     onehot_relations_classes.remove('has_time_info')
#     onehot_relations_classes.remove('is_located')
#     onehot_relations_classes.remove('is_specified')
    
    return onehot_relations, onehot_relations_classes

onehot_concepts, onehot_concepts_classes = create_onehot_concepts(mlb, 'concepts')
onehot_values, onehot_values_classes = create_onehot_values(mlb, 'attributes')
onehot_relations, onehot_relations_classes = create_onehot_relations(mlb, 'relations')



# making compatibility for converting the data into a dataframe
tmp_concepts_list = []
for i in onehot_concepts:
    tmp_concepts_list.append(list(i))
tmp_values_list = []
for i in onehot_values:
    tmp_values_list.append(list(i))
tmp_relations_list=[]
for i in onehot_relations:
    tmp_relations_list.append(list(i))




tmp_dict={'concepts_onehot':tmp_concepts_list,
          'values_onehot':tmp_values_list,
          'relations_onehot': tmp_relations_list}
df_update = pd.DataFrame.from_dict(tmp_dict)
df = pd.concat([df, df_update], axis = 1)

import pickle
df.to_pickle("/home/rusi02/research_paper/big_medilytics/data/df_cavr_v1.pkl")

class_list = [onehot_concepts_classes, onehot_values_classes, onehot_relations_classes]

with open('/home/rusi02/research_paper/big_medilytics/data/classes_v1.pkl', 'wb') as f:
    pickle.dump(class_list, f)




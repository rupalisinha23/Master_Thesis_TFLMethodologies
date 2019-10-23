# this script creates a dataframe by reading the tsv file

import sys
import re
import pandas as pd


def filter_data(item):
    item = item.rstrip('\n').split('\t')
    sentence = re.sub('[\[\]\'\,\->\<\>\{\}\-\(\)\?]', '', item[1])
    concepts_tmp = re.sub('[\[\]\'\,]', '', item[2]).split(' ')
    for idx, val in enumerate(concepts_tmp):
        if '-' in val:
            concepts_tmp[idx] = val.split('-')[1]
        else:
            concepts_tmp[idx] = '<UNK>'
    for i in concepts_tmp:
        if i not in concepts:
            concepts.append(i)
    attributes = re.sub('[\[\]\'\,]', '', item[3]).split(' ')
    attributes = list(zip(attributes[::2], attributes[1::2]))
    relations = re.sub('[\{\}\'\:\,\[\]]', '', item[4]).split(' ')
    relations = list(zip(*[iter(relations)] * 3))

    return sentence, concepts, attributes, relations


# main working area
#input_file = sys.argv[1]

s=[]
c=[]
a=[]
r=[]

with open('/home/rusi02/output_car.tsv','r+', encoding='utf-8') as f:
    data = f.readlines()
concepts = []
for item in data:
    sentence, concepts, attributes, relations = filter_data(item)
    s.append(sentence)
    c.append(concepts)
    a.append(attributes)
    r.append(relations)
    sentence= []
    concepts = []
    attributes = []
    relations = []
    dict_tmp = {'sentence': s,'concepts':c,'attributes':a,'relations':r}

df = pd.DataFrame.from_dict(dict_tmp)
print("Data frame is of {} size.".format(df.shape))
print(df['concepts'])

# code for creating the input file to fasttext to create .vec

#
# from textblob_de import TextBlobDE
# with open('/Users/rupalisinha/TransferNN/preprocess/Output_files/full_length_sentences.txt','a+') as f:
#     for i in df['sentence']:
#         blob = TextBlobDE(i).tokens
#         f.writelines(' '.join(blob) + u'\n')










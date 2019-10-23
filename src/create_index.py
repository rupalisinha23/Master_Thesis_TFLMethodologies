import sys


# returns a dictionary for mapping train, val, test sentence vocabulary

def map_index(path):
    word2idx_train = {'<PAD>': 0, '<UNK>': 1}
    word2idx_val = {'<PAD>': 0, '<UNK>': 1}
    word2idx_test = {'<PAD>': 0, '<UNK>': 1}
    count_train = 2
    count_val = 2
    count_test = 2
    with open(path,'r+', encoding='utf-8') as file:
        data = file.readlines()
    for item in data:
        word_train = item.split('\t')[0].strip('\n')
        word_val = item.split('\t')[1].strip('\n')
        word_test = item.split('\t')[2].strip('\n')
        if word_test != 'None':
            if word_test not in word2idx_test:
                word2idx_test[word_test] = count_test
                count_test +=1
        if word_val != 'None':
            if word_val not in word2idx_val:
                word2idx_val[word_val] = count_val
                count_val +=1
        if word2idx_train != 'None' or word2idx_train != '':
            if word_train not in word2idx_train:
                word2idx_train[word_train] = count_train
                count_train +=1

    return word2idx_train, word2idx_val,word2idx_test


path = sys.argv[1]
word2idx_train, word2idx_val, word2idx_test = map_index(path)


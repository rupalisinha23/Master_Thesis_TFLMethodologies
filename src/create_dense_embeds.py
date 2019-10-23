import pickle
import numpy as np
import sys


def create_concept_embeds(vocab_idx_weight_dict, concepts_tag):
    concepts_dict = {}
    for key, value in concepts_tag.items():
        interm = []
        for word in value:
            if word in vocab_idx_weight_dict:
                interm.append(vocab_idx_weight_dict[word][1])

            else:
                pass

        word_embed = np.mean(np.array(interm), axis=0)
        if key not in concepts_dict:
            concepts_dict[key] = word_embed
    
    
    return concepts_dict


def create_values_embeds(concepts_dict, values_tag, concepts_icf):
    # changing the embeddings of concept using icf factor
#     for key, value in concepts_dict.items():
#         concepts_dict[key] = value * concepts_icf[key]

        
    values_dict = {}
    for key, value in values_tag.items():
        em = []
        for concept_name in value:
            if concept_name in concepts_dict.keys():
                em.append(concepts_dict[concept_name])
        word_embed = np.mean(np.array(em), axis=0)
        if key not in values_dict:
            values_dict[key] = word_embed
    return values_dict
           


def create_relations_embeds(concepts_dict, relations_tag):
    relations_dict={}
    #iterate over relations_tag for key and value
    for key, value in relations_tag.items():
        interm = []
        for concept in value:
            if concept in concepts_dict:
                interm.append(concepts_dict[concept])
        word_embed = np.mean(np.array(interm), axis=0)
        if key not in relations_dict:
            relations_dict[key] = word_embed
    return relations_dict


def create_y_embeddings(vocab_idx_weight_dict, path_concepts, path_values, path_relations, concepts_icf):
    concepts_tag = pickle.load(open(path_concepts,"rb"))
    values_tag = pickle.load(open(path_values,"rb"))
    relations_tag = pickle.load(open(path_relations, "rb"))
    concepts_dict =  create_concept_embeds(vocab_idx_weight_dict, concepts_tag)
    values_dict = create_values_embeds(concepts_dict, values_tag, concepts_icf)
    relations_dict = create_relations_embeds(concepts_dict, relations_tag)
    return {'concepts_dict':concepts_dict,
            'values_dict': values_dict,
            'relations_dict': relations_dict}



    
    


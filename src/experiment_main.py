# import necessary libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import operator
import copy
import numpy as np
import pandas as pd
import logging
import sys
import os
import argparse
import numpy
import scipy
import pickle
from collections import OrderedDict
from VisdomLogger import VisdomLogger
import experiment_model as model
import experiment_record as rec
import RandomCVRecorder as ran
import create_dense_embeds as cde
import experiment_preprocess as ep


# passing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--vecfile')
parser.add_argument('--dffile')
parser.add_argument('--dffile_without_noise')
parser.add_argument('--classes')
parser.add_argument('--classes_without_noise')
parser.add_argument('--icf')
parser.add_argument('--concepts_dump')
parser.add_argument('--relations_dump')
parser.add_argument('--values_dump')
parser.add_argument('--flag', default="STL")
parser.add_argument('--plots', default="no")
parser.add_argument('--model', default="ft")
parser.add_argument('--trained_with_noise', default="yes")
parser.add_argument('--evaluate', default="without-noise")
parser.add_argument('--outputlayers')
parser.add_argument('--mtl_type')
args = parser.parse_args()

# pre-process the label string incase of MTL
# Example: concepts-relations-values --> ['concepts', 'relations', 'values']
outputlayers = args.outputlayers.split('-') if ('-')in args.outputlayers else args.outputlayers

if type(outputlayers) is not list:
    unique_label1 = 'values' if outputlayers == 'values' else 'relations' if outputlayers == 'relations' else 'concepts'
    assert unique_label1
    unique_labels = [unique_label1]
elif len(outputlayers) == 3:
    unique_label1 = 'values' if outputlayers[0] == 'values' else 'relations' if outputlayers[0] == 'relations' else 'concepts'
    unique_label2 = 'values' if outputlayers[1] == 'values' else 'relations' if outputlayers[1] == 'relations' else 'concepts'
    unique_label3 = 'values' if outputlayers[2] == 'values' else 'relations' if outputlayers[2] == 'relations' else 'concepts'
    assert unique_label1 != unique_label2 and unique_label2 != unique_label3 and unique_label1 != unique_label3
    unique_labels = [unique_label1, unique_label2, unique_label3]


# declare seed and set gpu/cpu
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#for logging info on the terminal
logging.getLogger().setLevel(logging.INFO)


# create the instance to the preprocess class
proc = ep.preprocess()

# import word2idx, embedding matrix by reading the .vec file
word2idx, weights, vocabulary_idx_weights = proc.read_vecfile_for_embeddings(args.vecfile)
# word2idx, weights = pickle.load(open("/home/rusi02/master_thesis/data/wikipedia_word2idx.pkl","rb")), pickle.load(open("/home/rusi02/master_thesis/data/wikipedia_weights.pkl","rb"))
logging.info(' Length of word2idx: {}'.format(len(word2idx)))


# split the data into train validation and test; transform the x data into list of word indices; pad the input data; convert the types and push the data to gpu

if args.trained_with_noise == "no":
    data = proc.create_data(args.dffile_without_noise, word2idx, args.classes_without_noise)
else:
    data = proc.create_data(args.dffile, word2idx, args.classes)


# converting the type of embeds
embeds_from_fasttext = torch.FloatTensor(weights)


# check the random search thing
pc = ran.ParamConfigGen()


# visdom logger 
# visl = VisdomLogger(environment_name='Thesis WORK '+ args.model +' trained with noise and evaluated with noise ' + args.flag + 'v1')


# store result tables 
exp_result_dir = '/home/rusi02/master_thesis/Output_files/Thesiswork_'+args.model+'test'+args.flag+'_v1.csv'
if type(outputlayers) != list:
    measures = ['ROC_AUC_MICRO_VAL'+unique_label1, 'ROC_AUC_MACRO_VAL'+unique_label1, 'ROC_AUC_MICRO_TEST'+unique_label1, 'ROC_AUC_MACRO_TEST'+unique_label1, 'F1_MICRO_VAL'+unique_label1, 'F1_MACRO_VAL'+unique_label1, 'F1_MICRO_TEST'+unique_label1,'F1_MACRO_TEST'+unique_label1, 'PRE_REC_MICRO_VAL'+unique_label1, 'PRE_REC_MACRO_VAL'+unique_label1, 'PRE_REC_MICRO_TEST'+unique_label1,'PRE_REC_MACRO_TEST'+unique_label1, 'LOSS_VAL','rand_val']
elif len(outputlayers) == 3:  
    measures = ['ROC_AUC_MICRO_VAL'+unique_label1, 'ROC_AUC_MACRO_VAL'+unique_label1, 
         'ROC_AUC_MICRO_TEST'+unique_label1, 'ROC_AUC_MACRO_TEST'+unique_label1, 
         'F1_MICRO_VAL'+unique_label1, 'F1_MACRO_VAL'+unique_label1,
         'F1_MICRO_TEST'+unique_label1, 'F1_MACRO_TEST'+unique_label1,
         'PRE_REC_MICRO_VAL'+unique_label1, 'PRE_REC_MACRO_VAL'+unique_label1, 
         'PRE_REC_MICRO_TEST'+unique_label1, 'PRE_REC_MACRO_TEST'+unique_label1,
         
         'ROC_AUC_MICRO_VAL'+unique_label2, 'ROC_AUC_MACRO_VAL'+unique_label2, 
         'ROC_AUC_MICRO_TEST'+unique_label2, 'ROC_AUC_MACRO_TEST'+unique_label2, 
         'F1_MICRO_VAL'+unique_label2, 'F1_MACRO_VAL'+unique_label2,
         'F1_MICRO_TEST'+unique_label2, 'F1_MACRO_TEST'+unique_label2,
         'PRE_REC_MICRO_VAL'+unique_label2, 'PRE_REC_MACRO_VAL'+unique_label2, 
         'PRE_REC_MICRO_TEST'+unique_label2, 'PRE_REC_MACRO_TEST'+unique_label2,
         
         'ROC_AUC_MICRO_VAL'+unique_label3, 'ROC_AUC_MACRO_VAL'+unique_label3, 
         'ROC_AUC_MICRO_TEST'+unique_label3, 'ROC_AUC_MACRO_TEST'+unique_label3, 
         'F1_MICRO_VAL'+unique_label3, 'F1_MACRO_VAL'+unique_label3,
         'F1_MICRO_TEST'+unique_label3, 'F1_MACRO_TEST'+unique_label3,
         'PRE_REC_MICRO_VAL'+unique_label3, 'PRE_REC_MACRO_VAL'+unique_label3, 
         'PRE_REC_MICRO_TEST'+unique_label3, 'PRE_REC_MACRO_TEST'+unique_label3,
         'LOSS_VAL'+unique_label1, 'LOSS_VAL'+unique_label2, 'LOSS_VAL'+unique_label3, 'LOSS_VAL_COMB', 'rand_val'] 

tasks = ['C']


# hyperparameters
params = OrderedDict([('model',[args.model]),
                      ('concepts_size',[len(data['y_train_concepts'][0])]),
                      ('values_size',[len(data['y_train_values'][0])]),
                      ('relations_size',[len(data['y_train_relations'][0])]),
                      ('vocab_size',[len(word2idx)]),
                      ('embedding_dims',[100]),
                      ('epochs',[10]),
                      ('batch_size',[100,300,50]),
                      ('learning_rate',[0.01, 0.02, 0.03, 0.04, 0.05, 0.015, 0.025]),
                      ('reruns',[5])   
                     ])


guarded_csv = ran.RecomputationGuardedCSVDictWriter(path=exp_result_dir,
                                                    params=params,
                                                    fields_to_ignore_in_hashing=["CV_run", 'fold'],
                                                    finished_fields=measures)


# declare the model object
if params['model'][0] == 'CNN':
    logging.info(' CNN model')
    
    if len(unique_labels)== 3:
        cnn_params = dict(flag=args.flag,vocab_size=len(word2idx), emb_dim=100, num_filters=150, filter_sizes=(1,2,3,4, 10), dropout=0, input_dropout=0, tune_embs=False, activation=torch.sigmoid)
        model = model.CNN(unique_labels, embeds_from_fasttext, cnn_params, params)
    elif len(unique_labels) == 1:
        cnn_params = dict(flag=args.flag,vocab_size=len(word2idx), emb_dim=100, num_filters=150, filter_sizes=(1,2,3,4, 10), dropout=0, input_dropout=0, tune_embs=False, activation=torch.sigmoid)
        model = model.CNN(unique_labels, embeds_from_fasttext, cnn_params, params)


elif params['model'][0] == 'FT':
    if type(outputlayers) != list:
        model = model.exp_tagger(args.flag,embeds_from_fasttext, params, outputlayers, args.mtl_type)
    elif len(outputlayers) == 3:
        model = model.exp_tagger(args.flag,embeds_from_fasttext, params, outputlayers, args.mtl_type)
           
    
if torch.cuda.device_count() > 1:
    logging.info(" Let's use {} GPUs! ".format(torch.cuda.device_count()))
model = nn.DataParallel(model)
model.to(device)
loss_function = nn.BCEWithLogitsLoss()


exp_num = 1


for x in pc.parameterConfigGenerator(params, random_search=True, CV=params['reruns'][0],  max_experiments = 2000,test_folds=[-1], verbose=True, random_seed=42, use_pytorch=True):
    logging.info(" Experiment_num: {}".format(exp_num))

    # use dataloader for batching the x and y --> for training data
    data_loader = DataLoader(dataset=data['x_train'], batch_size=x['batch_size'], shuffle=False, drop_last=True)
    data_loader_y_concepts = DataLoader(dataset=data['y_train_concepts'], batch_size=x['batch_size'], shuffle=False,
                                        drop_last=True)
    data_loader_y_values = DataLoader(dataset=data['y_train_values'], batch_size=x['batch_size'], shuffle=False,
                                        drop_last=True)
    data_loader_y_relations = DataLoader(dataset=data['y_train_relations'], batch_size=x['batch_size'], shuffle=False,
                                         drop_last=True)

    
    # declare the graph options to record
    # visdom_opts = rec.create_visdom_opts(outputlayers)
 
    
    if guarded_csv.check_if_already_run(run_params=x):
        """ Dont run an experiment twice """
        continue
    
    param_conf = rec.clean_dict(x)

    optimizer = optim.Adam(model.parameters(), lr=x['learning_rate'])
    
    # place to define all the variables to monitor in all the epochs
    best_model = {label: {'validation':{'micro':[0,0,0],
                                        'macro':[0,0,0],
                                       'f1_micro':[0,0,0],
                                       'f1_macro':[0,0,0],
                                       'pre_rec_micro':[0,0,0],
                                       'pre_rec_macro':[0,0,0]}, # [epoch_num, auc_score, y_preds]
                          'test'      :{'micro':[0,0,0],
                                        'macro':[0,0,0],
                                       'f1_micro':[0,0,0],
                                       'f1_macro':[0,0,0],
                                       'pre_rec_micro':[0,0,0],
                                       'pre_rec_macro':[0,0,0]}} for label in unique_labels
                 }


    roc_auc_per_epochs_micro1_val, roc_auc_per_epochs_macro1_val = [], []
    roc_auc_per_epochs_micro2_val, roc_auc_per_epochs_macro2_val = [], []
    roc_auc_per_epochs_micro3_val, roc_auc_per_epochs_macro3_val = [], []
    roc_auc_per_epochs_micro1_test, roc_auc_per_epochs_macro1_test = [], []
    roc_auc_per_epochs_micro2_test, roc_auc_per_epochs_macro2_test = [], []
    roc_auc_per_epochs_micro3_test, roc_auc_per_epochs_macro3_test = [], []
    
    f1_per_epochs_micro1_val, f1_per_epochs_macro1_val = [], []
    f1_per_epochs_micro1_test, f1_per_epochs_macro1_test = [], []
    f1_per_epochs_micro2_val, f1_per_epochs_macro2_val = [], []
    f1_per_epochs_micro2_test, f1_per_epochs_macro2_test = [], []
    f1_per_epochs_micro3_val, f1_per_epochs_macro3_val = [], []
    f1_per_epochs_micro3_test, f1_per_epochs_macro3_test = [], []
    
    pre_rec_per_epochs_micro1_val, pre_rec_per_epochs_macro1_val = [], []
    pre_rec_per_epochs_micro1_test, pre_rec_per_epochs_macro1_test = [], []
    pre_rec_per_epochs_micro2_val, pre_rec_per_epochs_macro2_val = [], []
    pre_rec_per_epochs_micro2_test, pre_rec_per_epochs_macro2_test = [], []
    pre_rec_per_epochs_micro3_val, pre_rec_per_epochs_macro3_val = [], []
    pre_rec_per_epochs_micro3_test, pre_rec_per_epochs_macro3_test = [], []
    

    # Training loop
    
    epoch_loss_list1, epoch_loss_list2, epoch_loss_list3, epoch_loss_comb =[],[],[],[]
    
    for epoch in range(0,x['epochs']):
        optimizer.zero_grad()

        step_size = 1      
        model.train()
        for batch, concepts_labels, value_labels, relations_labels in zip(data_loader, data_loader_y_concepts, data_loader_y_values, data_loader_y_relations):
            if (args.flag == 'STL'):

                binary_score1 = model.forward(batch)
                true1 = value_labels if unique_label1 == 'values' else relations_labels if unique_label1 == 'relations' else concepts_labels    
                batch_loss1 = loss_function(binary_score1, true1.to(device))
                batch_loss1.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss_list1.append(batch_loss1.cpu().data.numpy())
                binary_score1 = torch.sigmoid(binary_score1)
               
                
            elif (args.flag == 'MTL' and args.mtl_type == "A"):
                binary_score1, binary_score2, binary_score3 = model.forward(batch)
                true1 = value_labels if unique_label1 == 'values' else relations_labels if unique_label1 == 'relations' else concepts_labels 
                true2 = value_labels if unique_label2 == 'values' else relations_labels if unique_label2 == 'relations' else concepts_labels
                true3 = value_labels if unique_label3 == 'values' else relations_labels if unique_label3 == 'relations' else concepts_labels
                batch_loss1 = loss_function(binary_score1, true1.to(device))/3
                batch_loss2 = loss_function(binary_score2, true2.to(device))/3
                batch_loss3 = loss_function(binary_score3, true3.to(device))/3
                batch_loss_comb = (batch_loss1 + batch_loss2 + batch_loss3)
                batch_loss_comb.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss_list1.append(batch_loss1.cpu().data.numpy())
                epoch_loss_list2.append(batch_loss2.cpu().data.numpy())
                epoch_loss_list2.append(batch_loss3.cpu().data.numpy())
                epoch_loss_comb.append(batch_loss_comb.cpu().data.numpy())
                binary_score1 = torch.sigmoid(binary_score1)
                binary_score2 = torch.sigmoid(binary_score2)
                binary_score3 = torch.sigmoid(binary_score3)

            
        if (args.flag == 'STL'):
            epoch_loss = sum(epoch_loss_list1)/x['batch_size']
#             rec.call_visdom_options('TRAIN',outputlayers, args.mtl_type, visl, epoch, x, args.flag, epoch_loss) #TODO
#             logging.info(f'| Train BCELoss combined: {epoch_loss:.3f}')
        elif (args.flag == 'MTL' and args.mtl_type == "A"):
            epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss = sum(epoch_loss_list1)/x['batch_size'], sum(epoch_loss_list2)/x['batch_size'], sum(epoch_loss_list3)/x['batch_size'], sum(epoch_loss_comb)/x['batch_size']
            rec.call_visdom_options('TRAIN', outputlayers, args.mtl_type , visl,epoch, x, args.flag,epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss)
            logging.info(f'| Train BCELoss combined: {epoch_loss:.3f} | Train BCELoss label1: {epoch_loss1:.3f}| Train BCELoss label2: {epoch_loss2:.3f} | Train BCELoss label3: {epoch_loss3:.3f}')
                
                           
        # Validation starts
        
        model.eval()
        with torch.no_grad():
            if (args.flag == 'STL'):
                binary_score1_val = model.forward(data['x_val'])
                true1_val = data['y_val_'+unique_label1]
                epoch_loss1_val = loss_function(binary_score1_val, true1_val.to(device))
                binary_score1_val = torch.sigmoid(binary_score1_val)
                roc_auc_micro1, roc_auc_macro1, f1_micro1, f1_macro1, pre_rec_micro1, pre_rec_macro1 = rec.evaluate_using_roc_curve(len(true1_val.cpu().data.numpy()[0]), true1_val.cpu().data.numpy(), binary_score1_val.cpu().data.numpy(), data[unique_label1 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None,'validation')
                # to save the best model: store the ro auc scores

                roc_auc_per_epochs_micro1_val.append(roc_auc_micro1)
                roc_auc_per_epochs_macro1_val.append(roc_auc_macro1)
                
                f1_per_epochs_micro1_val.append(f1_micro1)
                f1_per_epochs_macro1_val.append(f1_macro1)
                
                pre_rec_per_epochs_micro1_val.append(pre_rec_micro1)
                pre_rec_per_epochs_macro1_val.append(pre_rec_macro1)
                
              
                
                best_model = rec.update_best_model('validation', x['epochs'], binary_score1_val.cpu().data.numpy(), best_model, unique_label1, roc_auc_per_epochs_micro1_val, roc_auc_per_epochs_macro1_val, f1_per_epochs_micro1_val, f1_per_epochs_macro1_val, pre_rec_per_epochs_micro1_val, pre_rec_per_epochs_macro1_val)

#                 rec.call_visdom_options('VALIDATION', outputlayers, args.mtl_type, visl, epoch, x, args.flag, epoch_loss1_val.cpu().data.numpy(), roc_auc_micro1, roc_auc_macro1, f1_micro1, f1_macro1, pre_rec_micro1, pre_rec_macro1)
                
                logging.info(f'| Validation loss label1: {epoch_loss1_val:.3f} | ROC MICRO Label1: {roc_auc_micro1:.3f} | ROC AUC MACRO Label1: {roc_auc_macro1:.3f} | F1 MICRO Label1: {f1_micro1:.3f} | F1 MACRO Label1: {f1_macro1:.3f} | Pre_rec MICRO Label1: {pre_rec_micro1:.3f} | Pre_rec MACRO Label1: {pre_rec_macro1:.3f} |')

                
                
            elif (args.flag == 'MTL' and args.mtl_type == "A"):
                binary_score1_val, binary_score2_val, binary_score3_val = model.forward(data['x_val'])
                true1_val, true2_val, true3_val = data['y_val_'+unique_label1], data['y_val_'+unique_label2], data['y_val_'+unique_label3]
                epoch_loss1_val = loss_function(binary_score1_val, true1_val.to(device))/3
                epoch_loss2_val = loss_function(binary_score2_val, true2_val.to(device))/3
                epoch_loss3_val = loss_function(binary_score3_val, true3_val.to(device))/3
                epoch_loss_comb_val = (epoch_loss1_val + epoch_loss2_val  + epoch_loss3_val)
                binary_score1_val = torch.sigmoid(binary_score1_val)
                binary_score2_val = torch.sigmoid(binary_score2_val)
                binary_score3_val = torch.sigmoid(binary_score3_val)

               
                roc_auc_micro1, roc_auc_macro1, f1_micro1, f1_macro1, pre_rec_micro1, pre_rec_macro1 = rec.evaluate_using_roc_curve(len(true1_val.cpu().data.numpy()[0]), true1_val.cpu().data.numpy(), binary_score1_val.cpu().data.numpy(), data[unique_label1 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'validation')
                roc_auc_micro2, roc_auc_macro2, f1_micro2, f1_macro2, pre_rec_micro2, pre_rec_macro2 = rec.evaluate_using_roc_curve(len(true2_val.cpu().data.numpy()[0]), true2_val.cpu().data.numpy(), binary_score2_val.cpu().data.numpy(), data[unique_label2 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'validation')
                roc_auc_micro3, roc_auc_macro3, f1_micro3, f1_macro3, pre_rec_micro3, pre_rec_macro3 = rec.evaluate_using_roc_curve(len(true3_val.cpu().data.numpy()[0]), true3_val.cpu().data.numpy(), binary_score3_val.cpu().data.numpy(), data[unique_label3 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'validation')

                
                # to save the best model: store the ro auc scores
                roc_auc_per_epochs_micro1_val.append(roc_auc_micro1)
                roc_auc_per_epochs_macro1_val.append(roc_auc_macro1)
                roc_auc_per_epochs_micro2_val.append(roc_auc_micro2)
                roc_auc_per_epochs_macro2_val.append(roc_auc_macro2)
                roc_auc_per_epochs_micro3_val.append(roc_auc_micro3)
                roc_auc_per_epochs_macro3_val.append(roc_auc_macro3)
                
                f1_per_epochs_micro1_val.append(f1_micro1)
                f1_per_epochs_macro1_val.append(f1_macro1)
                f1_per_epochs_micro2_val.append(f1_micro2)
                f1_per_epochs_macro2_val.append(f1_macro2)
                f1_per_epochs_micro3_val.append(f1_micro3)
                f1_per_epochs_macro3_val.append(f1_macro3)
                
                pre_rec_per_epochs_micro1_val.append(pre_rec_micro1)
                pre_rec_per_epochs_macro1_val.append(pre_rec_macro1)
                pre_rec_per_epochs_micro2_val.append(pre_rec_micro2)
                pre_rec_per_epochs_macro2_val.append(pre_rec_macro2)
                pre_rec_per_epochs_micro3_val.append(pre_rec_micro3)
                pre_rec_per_epochs_macro3_val.append(pre_rec_macro3)
                           
             
                best_model = rec.update_best_model('validation', x['epochs'], binary_score1_val.cpu().data.numpy(), best_model, unique_label1, roc_auc_per_epochs_micro1_val, roc_auc_per_epochs_macro1_val, f1_per_epochs_micro1_val, f1_per_epochs_macro1_val, pre_rec_per_epochs_micro1_val, pre_rec_per_epochs_macro1_val)
                best_model = rec.update_best_model('validation', x['epochs'], binary_score2_val.cpu().data.numpy(), best_model, unique_label2, roc_auc_per_epochs_micro2_val, roc_auc_per_epochs_macro2_val, f1_per_epochs_micro2_val,f1_per_epochs_macro2_val, pre_rec_per_epochs_micro2_val,pre_rec_per_epochs_macro2_val)
                best_model = rec.update_best_model('validation', x['epochs'], binary_score3_val.cpu().data.numpy(), best_model, unique_label3, roc_auc_per_epochs_micro3_val, roc_auc_per_epochs_macro3_val, f1_per_epochs_micro3_val, f1_per_epochs_macro3_val, pre_rec_per_epochs_micro3_val, pre_rec_per_epochs_macro3_val)
                
                rec.call_visdom_options('VALIDATION', outputlayers, args.mtl_type, visl, epoch, x, args.flag, epoch_loss1_val.cpu().data.numpy(), epoch_loss2_val.cpu().data.numpy(), epoch_loss3_val.cpu().data.numpy(), epoch_loss_comb_val.cpu().data.numpy(), roc_auc_micro1, roc_auc_macro1, roc_auc_micro2, roc_auc_macro2, roc_auc_micro3, roc_auc_macro3, f1_micro1, f1_macro1, f1_micro2, f1_macro2, f1_micro3, f1_macro3, pre_rec_micro1, pre_rec_macro1, pre_rec_micro2, pre_rec_macro2, pre_rec_micro3, pre_rec_macro3)
                
              
                logging.info(f'| Validation loss label1: {epoch_loss1_val:.3f} | Loss label2: {epoch_loss1_val:.3f} | ROC MICRO Label1: {roc_auc_micro1:.3f} | ROC AUC MACRO Label1: {roc_auc_macro1:.3f}| ROC MICRO Label2: {roc_auc_micro2:.3f} | ROC AUC MACRO Label2: {roc_auc_macro2:.3f}| ROC MICRO Label3: {roc_auc_micro3:.3f} | ROC AUC MACRO Label3: {roc_auc_macro3:.3f}| F1 MICRO Label1: {f1_micro1:.3f} | F1 MACRO Label1: {f1_macro1:.3f}| F1 MICRO Label2: {f1_micro2:.3f} | F1 MACRO Label2: {f1_macro2:.3f}| F1 MICRO Label3: {f1_micro3:.3f} | F1 MACRO Label3: {f1_macro3:.3f} | PRE_REC MICRO Label1: {pre_rec_micro1:.3f} | PRE_REC MACRO Label1: {pre_rec_macro1:.3f}| PRE_REC MICRO Label2: {pre_rec_micro2:.3f} | PRE_REC MACRO Label2: {pre_rec_macro2:.3f}| PRE_REC MICRO Label3: {pre_rec_micro3:.3f} | PRE_REC MACRO Label3: {pre_rec_macro3:.3f}')
                
                print('\n')
                
    
            
            ############### test loop starts #################
            
            if (args.flag == 'STL'):
                binary_score1_test = model.forward(data['x_test'])
                true1_test = data['y_test_'+unique_label1]
                epoch_loss1_test = loss_function(binary_score1_test, true1_test.to(device))
                binary_score1_test = torch.sigmoid(binary_score1_test)
                roc_auc_micro1_test, roc_auc_macro1_test, f1_micro1_test, f1_macro1_test, pre_rec_micro1_test, pre_rec_macro1_test = rec.evaluate_using_roc_curve(len(true1_test.cpu().data.numpy()[0]), true1_test.cpu().data.numpy(), binary_score1_test.cpu().data.numpy(), data[unique_label1 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'test')
                # to save the best model: store the ro auc scores

                roc_auc_per_epochs_micro1_test.append(roc_auc_micro1_test)
                roc_auc_per_epochs_macro1_test.append(roc_auc_macro1_test)
                f1_per_epochs_micro1_test.append(f1_micro1_test)
                f1_per_epochs_macro1_test.append(f1_macro1_test)
                pre_rec_per_epochs_micro1_test.append(pre_rec_micro1_test)
                pre_rec_per_epochs_macro1_test.append(pre_rec_macro1_test)
                
                best_model = rec.update_best_model('test', x['epochs'], binary_score1_test.cpu().data.numpy(), best_model, unique_label1, roc_auc_per_epochs_micro1_test, roc_auc_per_epochs_macro1_test, f1_per_epochs_micro1_test, f1_per_epochs_macro1_test, pre_rec_per_epochs_micro1_test, pre_rec_per_epochs_macro1_test)
                
                
                logging.info(f'| Test loss label1: {epoch_loss1_test:.3f} | Loss label2: {epoch_loss1_test:.3f} | ROC MICRO Label1: {roc_auc_micro1_test:.3f} | ROC AUC MACRO Label1: {roc_auc_macro1_test:.3f} | F1 MICRO Label1: {f1_micro1_test:.3f} | F1 MACRO Label1: {f1_macro1_test:.3f} | Pre_rec MICRO Label1: {pre_rec_micro1_test:.3f} | Pre_rec MACRO Label1: {pre_rec_macro1_test:.3f} |')
                

            elif (args.flag == 'MTL' and args.mtl_type == "A"):
                binary_score1_test, binary_score2_test, binary_score3_test = model.forward(data['x_test'])
                true1_test, true2_test, true3_test = data['y_test_'+unique_label1], data['y_test_'+unique_label2], data['y_test_'+unique_label3]
                epoch_loss1_test = loss_function(binary_score1_test, true1_test.to(device))/3
                epoch_loss2_test = loss_function(binary_score2_test, true2_test.to(device))/3
                epoch_loss3_test = loss_function(binary_score3_test, true3_test.to(device))/3
                epoch_loss_comb_test = (epoch_loss1_test + epoch_loss2_test + epoch_loss3_test)
                binary_score1_test = torch.sigmoid(binary_score1_test)
                binary_score2_test = torch.sigmoid(binary_score2_test)
                binary_score3_test = torch.sigmoid(binary_score3_test)
                
               
                roc_auc_micro1_test, roc_auc_macro1_test, f1_micro1_test, f1_macro1_test, pre_rec_micro1_test, pre_rec_macro1_test  = rec.evaluate_using_roc_curve(len(true1_test.cpu().data.numpy()[0]), true1_test.cpu().data.numpy(), binary_score1_test.cpu().data.numpy(), data[unique_label1 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'test')
                roc_auc_micro2_test, roc_auc_macro2_test, f1_micro2_test, f1_macro2_test, pre_rec_micro2_test, pre_rec_macro2_test  = rec.evaluate_using_roc_curve(len(true2_test.cpu().data.numpy()[0]), true2_test.cpu().data.numpy(), binary_score2_test.cpu().data.numpy(), data[unique_label2 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'test')
                roc_auc_micro3_test, roc_auc_macro3_test, f1_micro3_test, f1_macro3_test, pre_rec_micro3_test, pre_rec_macro3_test = rec.evaluate_using_roc_curve(len(true3_test.cpu().data.numpy()[0]), true3_test.cpu().data.numpy(), binary_score3_test.cpu().data.numpy(), data[unique_label3 + '_class'], args.plots, args.classes_without_noise, outputlayers, args.evaluate, None, 'test')
                
                
                # to save the best model: store the ro auc scores
                roc_auc_per_epochs_micro1_test.append(roc_auc_micro1_test)
                roc_auc_per_epochs_macro1_test.append(roc_auc_macro1_test)
                roc_auc_per_epochs_micro2_test.append(roc_auc_micro2_test)
                roc_auc_per_epochs_macro2_test.append(roc_auc_macro2_test)
                roc_auc_per_epochs_micro3_test.append(roc_auc_micro3_test)
                roc_auc_per_epochs_macro3_test.append(roc_auc_macro3_test)
                
                f1_per_epochs_micro1_test.append(f1_micro1_test)
                f1_per_epochs_macro1_test.append(f1_macro1_test)
                f1_per_epochs_micro2_test.append(f1_micro2_test)
                f1_per_epochs_macro2_test.append(f1_macro2_test)
                f1_per_epochs_micro3_test.append(f1_micro3_test)
                f1_per_epochs_macro3_test.append(f1_macro3_test)
                
                pre_rec_per_epochs_micro1_test.append(pre_rec_micro1_test)
                pre_rec_per_epochs_macro1_test.append(pre_rec_macro1_test)
                pre_rec_per_epochs_micro2_test.append(pre_rec_micro2_test)
                pre_rec_per_epochs_macro2_test.append(pre_rec_macro2_test)
                pre_rec_per_epochs_micro3_test.append(pre_rec_micro3_test)
                pre_rec_per_epochs_macro3_test.append(pre_rec_macro3_test)
                           

               
                best_model = rec.update_best_model('test', x['epochs'], binary_score1_test.cpu().data.numpy(), best_model, unique_label1, roc_auc_per_epochs_micro1_test, roc_auc_per_epochs_macro1_test, f1_per_epochs_micro1_test, f1_per_epochs_macro1_test, pre_rec_per_epochs_micro1_test, pre_rec_per_epochs_macro1_test )
                best_model = rec.update_best_model('test', x['epochs'], binary_score2_test.cpu().data.numpy(), best_model, unique_label2, roc_auc_per_epochs_micro2_test, roc_auc_per_epochs_macro2_test, f1_per_epochs_micro2_test, f1_per_epochs_macro2_test, pre_rec_per_epochs_micro2_test, pre_rec_per_epochs_macro2_test)
                best_model = rec.update_best_model('test', x['epochs'], binary_score3_test.cpu().data.numpy(), best_model, unique_label3, roc_auc_per_epochs_micro3_test, roc_auc_per_epochs_macro3_test, f1_per_epochs_micro3_test, f1_per_epochs_macro3_test, pre_rec_per_epochs_micro3_test, pre_rec_per_epochs_macro3_test)
                
              
                logging.info(f'| Test loss label1: {epoch_loss1_test:.3f} | Loss label2: {epoch_loss1_test:.3f} | ROC MICRO Label1: {roc_auc_micro1_test:.3f} | ROC AUC MACRO Label1: {roc_auc_macro1_test:.3f}| ROC MICRO Label2: {roc_auc_micro2_test:.3f} | ROC AUC MACRO Label2: {roc_auc_macro2_test:.3f}| ROC MICRO Label3: {roc_auc_micro3_test:.3f} | ROC AUC MACRO Label3: {roc_auc_macro3_test:.3f}| F1 MICRO Label1: {f1_micro1_test:.3f} | F1 MACRO Label1: {f1_macro1_test:.3f}| F1 MICRO Label2: {f1_micro2_test:.3f} | F1 MACRO Label2: {f1_macro2_test:.3f}| F1 MICRO Label3: {f1_micro3_test:.3f} | F1 MACRO Label3: {f1_macro3_test:.3f}| PRE_REC MICRO Label1: {pre_rec_micro1_test:.3f} | PRE_REC MACRO Label1: {pre_rec_macro1_test:.3f}| PRE_REC MICRO Label2: {pre_rec_micro2_test:.3f} | PRE_REC MACRO Label2: {pre_rec_macro2_test:.3f}| PRE_REC MICRO Label3: {pre_rec_micro3_test:.3f} | PRE_REC MACRO Label3: {pre_rec_macro3_test:.3f}|')
                print('\n')
            

    # record results
    for label in unique_labels:
        best_model[label]['validation']['trues'] = data['y_val_'+label].cpu().data.numpy() 
        best_model[label]['test']['trues'] = data['y_test_'+label].cpu().data.numpy()
    

    path_base = '/home/rusi02/master_thesis/best_model'
    path_mid = args.flag
    if args.flag == "MTL":
        path_end = "type"+args.mtl_type
        path1 = "TnoiseEnoise"
    elif args.flag == "STL":
        path_end = unique_label1
        path1 = "TnoiseEnoise"

    
    
    path_name='FT_0th_epoch_' + str(exp_num) + '_' +'v1'+ '.pickle'
    with open(os.path.join(path_base, path_mid, path_end, path1, path_name),'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    exp_num += 1    


    if type(outputlayers) != list:
        param_conf['ROC_AUC_MICRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['micro'][1],
        param_conf['ROC_AUC_MACRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['macro'][1],
        param_conf['ROC_AUC_MICRO_TEST'+unique_label1] = best_model[unique_label1]['test']['micro'][1],
        param_conf['ROC_AUC_MACRO_TEST'+unique_label1] = best_model[unique_label1]['test']['macro'][1],
        
        param_conf['F1_MICRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['f1_micro'][1],
        param_conf['F1_MACRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['f1_macro'][1],
        param_conf['F1_MICRO_TEST'+unique_label1] = best_model[unique_label1]['test']['f1_micro'][1],
        param_conf['F1_MACRO_TEST'+unique_label1] = best_model[unique_label1]['test']['f1_macro'][1],
        
        param_conf['PRE_REC_MICRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['pre_rec_micro'][1],
        param_conf['PRE_REC_MACRO_VAL'+unique_label1] = best_model[unique_label1]['validation']['pre_rec_macro'][1],
        param_conf['PRE_REC_MICRO_TEST'+unique_label1] = best_model[unique_label1]['test']['pre_rec_micro'][1],
        param_conf['PRE_REC_MACRO_TEST'+unique_label1] = best_model[unique_label1]['test']['pre_rec_macro'][1],
        
        
        param_conf['LOSS_VAL'] = epoch_loss1_val.item(),
        param_conf['rand_val']= x['rand_val']

        
        
        
    elif len(outputlayers) == 3:
        param_conf['LOSS_VAL_COMB'] = epoch_loss_comb_val.item()
        param_conf['rand_val'] = x['rand_val']
        for label in unique_labels:
            param_conf['ROC_AUC_MICRO_VAL'+label] = best_model[label]['validation']['micro'][1],
            param_conf['ROC_AUC_MACRO_VAL'+label] = best_model[label]['validation']['macro'][1],
            param_conf['ROC_AUC_MICRO_TEST'+label] = best_model[label]['test']['micro'][1],
            param_conf['ROC_AUC_MACRO_TEST'+label] = best_model[label]['test']['macro'][1],
            
            param_conf['F1_MICRO_VAL'+label] = best_model[label]['validation']['f1_micro'][1],
            param_conf['F1_MACRO_VAL'+label] = best_model[label]['validation']['f1_macro'][1],
            param_conf['F1_MICRO_TEST'+label] = best_model[label]['test']['f1_micro'][1],
            param_conf['F1_MACRO_TEST'+label] = best_model[label]['test']['f1_macro'][1],
            
            param_conf['PRE_REC_MICRO_VAL'+label] = best_model[label]['validation']['pre_rec_micro'][1],
            param_conf['PRE_REC_MACRO_VAL'+label] = best_model[label]['validation']['pre_rec_macro'][1],
            param_conf['PRE_REC_MICRO_TEST'+label] = best_model[label]['test']['pre_rec_micro'][1],
            param_conf['PRE_REC_MACRO_TEST'+label] = best_model[label]['test']['pre_rec_macro'][1],
            
            param_conf['LOSS_VAL'+label] = epoch_loss1_val.item()
            

    guarded_csv.writerow(param_conf)
                                          

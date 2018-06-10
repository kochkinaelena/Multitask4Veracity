#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:57:40 2017

@author: Helen
"""
import os
from keras.models import Model
from keras.layers import Input,LSTM, Dense, Masking, Dropout, TimeDistributed
from keras import regularizers
import numpy as np
import pickle
from hyperopt import STATUS_OK
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences
#%%
def build_model(params, num_features):
    
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int (params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    l2reg = params['l2reg']

    inputs_ab = Input(shape=(None,num_features))
    mask_ab = Masking(mask_value=0.)(inputs_ab)
    lstm_ab = LSTM(num_lstm_units, return_sequences=True)(mask_ab)
    for nl in range(num_lstm_layers-1): 
        lstm_ab2 = LSTM(num_lstm_units, return_sequences=True)(lstm_ab)
        lstm_ab = lstm_ab2
#    lstma = LSTM(num_lstm_units, return_sequences=True)(lstm_ab)    
    hidden1_a = TimeDistributed(Dense(num_dense_units))(lstm_ab)
    for nl in range(num_dense_layers-1):
        hidden2_a = TimeDistributed(Dense(num_dense_units))(hidden1_a)
        hidden1_a = hidden2_a
    dropout_a = Dropout(0.5)(hidden1_a)
    softmax_a = TimeDistributed(
                    Dense(4, activation='softmax',
                          activity_regularizer=regularizers.l2(l2reg)))(
                                                                  dropout_a)   
    #slice_layer=Lambda(lambda x:x[:,-1,:],output_shape=lambda s:(s[0],s[2]))
    #sliced = slice_layer(lstm_b)
    lstm_b = LSTM(num_lstm_units, return_sequences=False)(lstm_ab)
#    y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
#    print slice_layer.output_shape
    hidden1_b = Dense(num_dense_units)(lstm_b)
    for nl in range(num_dense_layers-1):
        hidden2_b = Dense(num_dense_units)(hidden1_b)
        hidden1_b = hidden2_b
    dropout_b = Dropout(0.5)(hidden1_b)
    softmax_b = Dense(3, activation='softmax',
                      activity_regularizer=regularizers.l2(l2reg))(dropout_b)
    model = Model(inputs=inputs_ab, outputs=[softmax_a, softmax_b])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def training (params,x_train,y_trainA,y_trainB):
    num_epochs = params['num_epochs'] 
    batchsize = params['batchsize']
    num_features = np.shape(x_train)[2]
    model = build_model(params, num_features)
    model.fit(x_train, [y_trainA, y_trainB],
          epochs=num_epochs, batch_size=batchsize, verbose=0)
                
    return  model


def objective_MTL2_RumEval(params):
    
    path = 'saved_data/saved_data_RumEv'

    x_train = np.load(os.path.join(path,
                                   'train/train_array.npy'))
    y_trainA = np.load(os.path.join(path,
                                    'train/fold_stance_labels.npy'))
    y_trainB = np.load(os.path.join(path,
                                    'train/labels.npy'))
    y_trainB = to_categorical(y_trainB, num_classes=3)
    y_train_cat = y_trainA    
    x_test = np.load ( os.path.join(path,
                                    'dev/train_array.npy'))
    y_testA = np.load(os.path.join(path,
                                   'dev/fold_stance_labels.npy'))
    y_testB = np.load ( os.path.join(path,
                                     'dev/labels.npy'))
    ids_testA = np.load ( os.path.join(path,
                                       'dev/tweet_ids.npy'))
    ids_testB = np.load ( os.path.join(path,
                                       'dev/ids.npy'))

    model = training(params, x_train,y_train_cat, y_trainB)
    pred_probabilities_a, pred_probabilities_b  = model.predict(x_test,
                                                                verbose=0)
    Y_pred_a = np.argmax(pred_probabilities_a, axis=2)
    Y_pred_b = np.argmax(pred_probabilities_b, axis=1)
    trees, tree_prediction, tree_label = branch2treelabels(ids_testB,
                                                           y_testB,
                                                           Y_pred_b)
    mactest_F_b = f1_score(tree_label, tree_prediction, average='macro')
    fids_test = ids_testA.flatten()
    fy_pred = Y_pred_a.flatten()
    Y_test_A = np.argmax(y_testA, axis=2)
    fy_test = Y_test_A.flatten()
    uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
    uniqtwid = uniqtwid.tolist()
    uindices2 = uindices2.tolist()
    del uniqtwid[0]
    del uindices2[0]
    uniq_dev_prediction = [fy_pred[i] for i in uindices2]
    uniq_dev_label = [fy_test[i] for i in uindices2]
    mactest_F_a = f1_score(uniq_dev_prediction,
                           uniq_dev_label,
                           average='macro')
    output = {'loss': (1-mactest_F_a)+(1-mactest_F_b),
              'Params': params,
              'status': STATUS_OK}
    return output
#%%

def eval_MTL2_RumEval(params, fname):
       
    path = 'saved_data/saved_data_RumEv'
    
    x_train = np.load (os.path.join(path, 'train/train_array.npy'))
    y_trainA = np.load(os.path.join(path, 'train/fold_stance_labels.npy'))
    y_trainB = np.load(os.path.join(path, 'train/labels.npy'))
    y_trainB = to_categorical(y_trainB, num_classes=3)
    
    x_dev = np.load ( os.path.join(path, 'dev/train_array.npy'))
    y_devA = np.load(os.path.join(path, 'dev/fold_stance_labels.npy'))
    y_devB = np.load(os.path.join(path, 'dev/labels.npy'))
    y_devB = to_categorical(y_devB, num_classes=3)
    
    x_test = np.load ( os.path.join(path, 'test/train_array.npy'))
    y_testA = np.load(os.path.join(path, 'test/fold_stance_labels.npy'))
    y_testB = np.load(os.path.join(path, 'test/labels.npy'))
    
    ids_testA = np.load ( os.path.join(path, 'test/tweet_ids.npy'))
    ids_testB = np.load ( os.path.join(path, 'test/ids.npy'))
    
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    y_devA = pad_sequences(y_devA, maxlen=len(y_trainA[0]), dtype='float32',
                           padding='post', truncating='post', value=0.)
    
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_trainA = np.concatenate((y_trainA, y_devA), axis=0)
    y_trainB = np.concatenate((y_trainB, y_devB), axis=0)
    y_train_cat = y_trainA
    
    model = training(params, x_train,y_train_cat,y_trainB)
    
    pred_probabilities_a, pred_probabilities_b  = model.predict(x_test,
                                                                verbose=0)
         
    Y_pred_a = np.argmax(pred_probabilities_a, axis=2)
    Y_pred_b = np.argmax(pred_probabilities_b, axis=1)
    
    trees, tree_prediction, tree_label = branch2treelabels(ids_testB,
                                                           y_testB,
                                                           Y_pred_b)
    
    Bmactest_P, Bmactest_R, Bmactest_F, _ = precision_recall_fscore_support(
                                                tree_label,
                                                tree_prediction,
                                                average='macro')    
    Bmictest_P, Bmictest_R, Bmictest_F, _ = precision_recall_fscore_support(
                                                tree_label,
                                                tree_prediction,
                                                average='micro')    
    Btest_P, Btest_R, Btest_F, _ = precision_recall_fscore_support(
                                        tree_label,
                                        tree_prediction)    
    Bacc = accuracy_score(tree_label, tree_prediction)
    fids_test = ids_testA.flatten()
    fy_pred = Y_pred_a.flatten()
    Y_test_A = np.argmax(y_testA, axis=2)
    fy_test = Y_test_A.flatten()
    uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
    uniqtwid = uniqtwid.tolist()
    uindices2 = uindices2.tolist()
    del uniqtwid[0]
    del uindices2[0]
    uniq_dev_prediction =  [fy_pred[i] for i in uindices2]
    uniq_dev_label =  [fy_test[i] for i in uindices2]
    
    Amactest_P, Amactest_R, Amactest_F, _ = precision_recall_fscore_support(
                                                uniq_dev_label,
                                                uniq_dev_prediction,
                                                average='macro')    
    Amictest_P, Amictest_R, Amictest_F, _ = precision_recall_fscore_support(
                                                uniq_dev_label,
                                                uniq_dev_prediction,
                                                average='micro')    
    Atest_P, Atest_R, Atest_F, _ = precision_recall_fscore_support(
                                        uniq_dev_label,
                                        uniq_dev_prediction)    
    Aacc = accuracy_score(uniq_dev_label, uniq_dev_prediction)
    
    output = {  
              'Params': params,
              'TaskA':{
                      'accuracy': Aacc,
                      'Macro': {'Macro_Precision': Amactest_P,
                                'Macro_Recall': Amactest_R,
                                'Macro_F_score': Amactest_F},
                      'Micro': {'Micro_Precision': Amictest_P,
                                'Micro_Recall': Amictest_R,
                                'Micro_F_score': Amictest_F}, 
                      'Per_class': {'Pclass_Precision': Atest_P,
                                    'Pclass_Recall': Atest_R,
                                    'Pclass_F_score': Atest_F}
                      },
              'TaskB':{
                      'accuracy': Bacc,
                      'Macro': {'Macro_Precision': Bmactest_P,
                                'Macro_Recall': Bmactest_R,
                                'Macro_F_score': Bmactest_F},
                      'Micro': {'Micro_Precision': Bmictest_P,
                                'Micro_Recall': Bmictest_R,
                                'Micro_F_score': Bmictest_F}, 
                      'Per_class': {'Pclass_Precision': Btest_P,
                                    'Pclass_Recall': Btest_R,
                                    'Pclass_F_score': Btest_F}
                      },
              'attachments': {'Task A':{'ID': uniqtwid,
                                        'Label': uniq_dev_label,
                                        'Prediction': uniq_dev_prediction},
                              'Task B': {'ID': trees,
                                         'Label':tree_label,
                                         'Prediction': tree_prediction,
                                         'Branch': {'ID': ids_testB,
                                                    'Label': y_testB,
                                                    'Prediction': Y_pred_b} 
                                        }
                     }
             }    
    directory = "output"
    if not os.path.exists(directory):
        os.mkdir(directory)

    with open('output/output'+fname+'.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)   

    return output

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
from copy import deepcopy
from itertools import compress
#%%
    
def build_model(params, num_features):
    
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int (params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    l2reg = params['l2reg']
    
    inputs_ab = Input(shape=(None,num_features))
    mask_ab = Masking(mask_value=0.)(inputs_ab)
    lstm_abc = LSTM(num_lstm_units, return_sequences=True)(mask_ab)
    for nl in range(num_lstm_layers-1): 
        
        lstm_ab2 = LSTM(num_lstm_units, return_sequences=True)(lstm_abc)
        lstm_abc = lstm_ab2

#    lstma =LSTM(num_lstm_units, return_sequences=True)(lstm_abc)     
    hidden1_a = TimeDistributed(Dense(num_dense_units))(lstm_abc)
    for nl in range(num_dense_layers-1):
    
        hidden2_a = TimeDistributed(Dense(num_dense_units))(hidden1_a)
        hidden1_a = hidden2_a
    dropout_a = Dropout(0.5)(hidden1_a)
    softmax_a = TimeDistributed(
                    Dense(4,
                          activation='softmax',
                          activity_regularizer=regularizers.l2(l2reg)),
                          name='softmaxa')(dropout_a)
    
    lstm_b = LSTM(num_lstm_units, return_sequences=False)(lstm_abc)
    hidden1_b = Dense(num_dense_units)(lstm_b)
    for nl in range(num_dense_layers-1):
        hidden2_b = Dense(num_dense_units)(hidden1_b)
        hidden1_b = hidden2_b
    dropout_b = Dropout(0.5)(hidden1_b)
    softmax_b = Dense(3, activation='softmax',
                      activity_regularizer=regularizers.l2(l2reg),
                      name='softmaxb')(dropout_b)
    
    lstm_c = LSTM(num_lstm_units, return_sequences=False)(lstm_abc)
    hidden1_c = Dense(num_dense_units)(lstm_c)
    for nl in range(num_dense_layers-1):
        hidden2_c = Dense(num_dense_units)(hidden1_c)
        hidden1_c = hidden2_c
    dropout_c = Dropout(0.5)(hidden1_c)
    softmax_c = Dense(2, activation='softmax',
                      activity_regularizer=regularizers.l2(l2reg),
                      name='softmaxc')(dropout_c)
    
    model = Model(inputs=inputs_ab, outputs=[softmax_a, softmax_b, softmax_c])
    
    model.compile(optimizer='adam', 
                  loss={'softmaxa': 'categorical_crossentropy',
                        'softmaxb': 'categorical_crossentropy',
                        'softmaxc': 'binary_crossentropy' },
                  loss_weights={'softmaxa': 0.3,
                                'softmaxb': 0.4,
                                'softmaxc': 0.3},
                  metrics=['accuracy'],
                  sample_weight_mode={'softmaxa': 'temporal',
                                      'softmaxb': None,
                                      'softmaxc': None})
     
    return model


def training (params,x_train,y_trainA,y_trainB,y_trainC):
    num_epochs = params['num_epochs'] 
    batchsize = params['batchsize']
    
    maskA = np.any(y_trainA,axis=2) + 0.00001
    maskB = np.any(y_trainB,axis=1) + 0.00001

    num_features = np.shape(x_train)[2]
    model = build_model(params, num_features)
    
    model.fit(x_train, {"softmaxa":y_trainA,"softmaxb": y_trainB,
                        "softmaxc": y_trainC}, 
              sample_weight={"softmaxa":maskA, "softmaxb": maskB,
                             "softmaxc": None},
              epochs=num_epochs, batch_size=batchsize,verbose=0)
                
    return  model
#%%
def objective_MTL3_CV5(params):    
    path = 'saved_data/saved_data_MTL3'
   
    train = ['ferguson', 
             'ottawashooting', 
             'sydneysiege', 
            ]
    
    test = 'charliehebdo' #'germanwings-crash' 

    max_branch_len = 25
    x_train = []
    ya_train = []
    yb_train = []
    yc_train = []

    for t in train:
        temp_x_train = np.load(os.path.join(path,t, 'train_array.npy'))
        temp_ya_train = np.load(os.path.join(path,t,
                                             'fold_stance_labels.npy'))
        temp_yb_train = np.load(os.path.join(path,t, 'labels.npy'))
        temp_yc_train = np.load(os.path.join(path,t, 'rnr_labels.npy'))
        
        # pad sequences to the size of the largest 
        temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len,
                                     dtype='float32', padding='post',
                                     truncating='post', value=0.)
        temp_ya_train = pad_sequences(temp_ya_train, maxlen=max_branch_len,
                                      dtype='int32', padding='post',
                                      truncating='post', value=0.)
        
        x_train.extend(temp_x_train)
        ya_train.extend(temp_ya_train)
        yb_train.extend(temp_yb_train)
        yc_train.extend(temp_yc_train)   
    
    x_train = np.asarray(x_train)
    ya_train = np.asarray(ya_train)
    yb_train = np.asarray(yb_train)
    yc_train = np.asarray(yc_train)
    yc_train = to_categorical(yc_train, num_classes=2)
    x_test = np.load(os.path.join(path,test, 'train_array.npy'))
    ya_test = np.load(os.path.join(path,test, 'fold_stance_labels.npy'))
    yb_test = np.load(os.path.join(path,test, 'labels.npy'))
    yc_test = np.load(os.path.join(path,test, 'rnr_labels.npy'))
    ids_testA = np.load(os.path.join(path, test, 'tweet_ids.npy'))
    ids_testBC = np.load(os.path.join(path, test, 'ids.npy'))
    
    
    model = training(params, x_train, ya_train, yb_train, yc_train)
   
    pred_probabilities_a, pred_probabilities_b, pred_probabilities_c = model.predict(x_test,verbose=0)

   
    Y_pred_a = np.argmax(pred_probabilities_a, axis=2)
    Y_pred_b = np.argmax(pred_probabilities_b, axis=1)
    Y_pred_c = np.argmax(pred_probabilities_c, axis=1)
    
    maskB = np.any(yb_test, axis=1)
    ids_testB = list(compress(ids_testBC, maskB))
    yb_test = list(compress(yb_test, maskB))
    Y_pred_b = list(compress(Y_pred_b, maskB))

    ids_testB = np.asarray(ids_testB)
    yb_test = np.asarray(yb_test)
    Y_pred_b = np.asarray(Y_pred_b)
    yb_test = np.argmax(yb_test, axis=1)

    trees, tree_prediction, tree_label = branch2treelabels(ids_testB,
                                                           yb_test,
                                                           Y_pred_b)
    
    treesC, tree_predictionC, tree_labelC = branch2treelabels(ids_testBC,
                                                              yc_test,
                                                              Y_pred_c)

    mactest_F_b = f1_score(tree_label, tree_prediction, average='macro',
                           labels=[0,1,2])
    mactest_F_c = f1_score(tree_labelC, tree_predictionC, average='binary')
    
    maskA = np.any(np.any(ya_test,axis=2),axis=1)
    
    if np.any(maskA):
    
        Y_true_a = ya_test[maskA]
        Y_true_a = np.argmax(Y_true_a, axis=2)
        Y_pred_a = Y_pred_a[maskA]
        ids_testA = ids_testA[maskA]
       
        fids_test = ids_testA.flatten()
        fy_pred = Y_pred_a.flatten()
        fy_test = Y_true_a.flatten()
        
        uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
        uniqtwid = uniqtwid.tolist()
        uindices2 = uindices2.tolist()
        del uindices2[uniqtwid.index(b'a')]
        del uniqtwid[uniqtwid.index(b'a')]
        
        uniq_dev_prediction = [fy_pred[i] for i in uindices2]
        uniq_dev_label =  [fy_test[i] for i in uindices2]
        
        mactest_F_a = f1_score(uniq_dev_prediction, uniq_dev_label,
                               average='macro',labels=[0,1,2,3])
    else:
        mactest_F_a = 0
        uniq_dev_prediction = []
        uniq_dev_label = []

    output = {'loss': (1-mactest_F_a)+(1-mactest_F_b)+(1-mactest_F_c),
              'Params': params,
              'status': STATUS_OK,
             }
   
    return output
    
def objective_MTL3_CV9(params):    
    path = 'saved_data/saved_data_MTL3'
   
    train = ['ferguson', 'ottawashooting','sydneysiege', 'putinmissing',
             'prince-toronto', 'gurlitt', 'ebola-essien']    
    test =  'charliehebdo'

    max_branch_len = 25
    x_train = []
    ya_train = []
    yb_train = []
    yc_train = []

    for t in train:
        temp_x_train = np.load(os.path.join(path,t, 'train_array.npy'))
        temp_ya_train = np.load(os.path.join(path,t,
                                             'fold_stance_labels.npy'))
        temp_yb_train = np.load(os.path.join(path,t, 'labels.npy'))
        temp_yc_train = np.load(os.path.join(path,t, 'rnr_labels.npy'))
        
        # pad sequences to the size of the largest 
        temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len,
                                     dtype='float32', padding='post',
                                     truncating='post', value=0.)
        temp_ya_train = pad_sequences(temp_ya_train, maxlen=max_branch_len,
                                      dtype='int32', padding='post',
                                      truncating='post', value=0.)
        
        x_train.extend(temp_x_train)
        ya_train.extend(temp_ya_train)
        yb_train.extend(temp_yb_train)
        yc_train.extend(temp_yc_train)   
    
    x_train = np.asarray(x_train)
    ya_train = np.asarray(ya_train)
    yb_train = np.asarray(yb_train)
    yc_train = np.asarray(yc_train)
    yc_train = to_categorical(yc_train, num_classes=2)
    x_test = np.load(os.path.join(path,test, 'train_array.npy'))
    ya_test = np.load(os.path.join(path,test, 'fold_stance_labels.npy'))
    yb_test = np.load(os.path.join(path,test, 'labels.npy'))
    yc_test = np.load(os.path.join(path,test, 'rnr_labels.npy'))
    ids_testA = np.load(os.path.join(path, test, 'tweet_ids.npy'))
    ids_testBC = np.load(os.path.join(path, test, 'ids.npy'))
    
    
    model = training(params, x_train, ya_train, yb_train, yc_train)
   
    pred_probabilities_a, pred_probabilities_b, pred_probabilities_c = model.predict(x_test,verbose=0)

   
    Y_pred_a = np.argmax(pred_probabilities_a, axis=2)
    Y_pred_b = np.argmax(pred_probabilities_b, axis=1)
    Y_pred_c = np.argmax(pred_probabilities_c, axis=1)
    
    maskB = np.any(yb_test, axis=1)
    ids_testB = list(compress(ids_testBC, maskB))
    yb_test = list(compress(yb_test, maskB))
    Y_pred_b = list(compress(Y_pred_b, maskB))

    ids_testB = np.asarray(ids_testB)
    yb_test = np.asarray(yb_test)
    Y_pred_b = np.asarray(Y_pred_b)
    yb_test = np.argmax(yb_test, axis=1)

    trees, tree_prediction, tree_label = branch2treelabels(ids_testB,
                                                           yb_test,
                                                           Y_pred_b)
    
    treesC, tree_predictionC, tree_labelC = branch2treelabels(ids_testBC,
                                                              yc_test,
                                                              Y_pred_c)

    mactest_F_b = f1_score(tree_label, tree_prediction, average='macro',
                           labels=[0,1,2])
    mactest_F_c = f1_score(tree_labelC, tree_predictionC, average='binary')
    
    maskA = np.any(np.any(ya_test,axis=2),axis=1)
    
    if np.any(maskA):
    
        Y_true_a = ya_test[maskA]
        Y_true_a = np.argmax(Y_true_a, axis=2)
        Y_pred_a = Y_pred_a[maskA]
        ids_testA = ids_testA[maskA]
       
        fids_test = ids_testA.flatten()
        fy_pred = Y_pred_a.flatten()
        fy_test = Y_true_a.flatten()
        
        uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
        uniqtwid = uniqtwid.tolist()
        uindices2 = uindices2.tolist()
        del uindices2[uniqtwid.index(b'a')]
        del uniqtwid[uniqtwid.index(b'a')]
        
        uniq_dev_prediction = [fy_pred[i] for i in uindices2]
        uniq_dev_label =  [fy_test[i] for i in uindices2]
        
        mactest_F_a = f1_score(uniq_dev_prediction, uniq_dev_label,
                               average='macro',labels=[0,1,2,3])
    else:
        mactest_F_a = 0
        uniq_dev_prediction = []
        uniq_dev_label = []

    output = {'loss': (1-mactest_F_a)+(1-mactest_F_b)+(1-mactest_F_c),
              'Params': params,
              'status': STATUS_OK,
             }
   
    return output    


def eval_MTL3(params, data, fname):
       
    path = 'saved_data/saved_data_MTL3'

    if data=='PHEME5':
        folds = ['charliehebdo', 'germanwings-crash', 'ferguson',
                 'ottawashooting', 'sydneysiege']
    else:
        folds = ['charliehebdo', 'germanwings-crash', 'ferguson',
                 'ottawashooting', 'sydneysiege', 'putinmissing',
                 'prince-toronto', 'gurlitt', 'ebola-essien']
    
    allfolds = []
    
    cv_ids_b = []
    cv_prediction_b = []
    cv_label_b = []
    
    cv_ids_a = []
    cv_prediction_a = []
    cv_label_a = []
    
    cv_ids_c = []
    cv_prediction_c = []
    cv_label_c = []
   
    for number in range(len(folds)): 
        
        test = folds[number]
        train = deepcopy(folds)
        del train[number]
   
        max_branch_len = 25
        x_train = []
        ya_train = []
        yb_train = []
        yc_train = []
         
        for t in train:
            temp_x_train = np.load(os.path.join(path,t, 'train_array.npy'))
            temp_ya_train = np.load(os.path.join(path,t,
                                                 'fold_stance_labels.npy'))
            temp_yb_train = np.load(os.path.join(path,t, 'labels.npy'))
            temp_yc_train = np.load(os.path.join(path,t, 'rnr_labels.npy'))
            
            # pad sequences to the size of the largest 
            temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len,
                                         dtype='float32', padding='post',
                                         truncating='post', value=0.)
            temp_ya_train = pad_sequences(temp_ya_train,
                                          maxlen=max_branch_len,
                                          dtype='int32',
                                          padding='post',
                                          truncating='post',
                                          value=0.)
            
            x_train.extend(temp_x_train)
            ya_train.extend(temp_ya_train)
            yb_train.extend(temp_yb_train)
            yc_train.extend(temp_yc_train)   
        
        x_train = np.asarray(x_train)
        ya_train = np.asarray(ya_train)
        yb_train = np.asarray(yb_train)
        yc_train = np.asarray(yc_train)
        yc_train = to_categorical(yc_train, num_classes=2)
        
        x_test = np.load(os.path.join(path,test, 'train_array.npy'))
        ya_test = np.load(os.path.join(path,test, 'fold_stance_labels.npy'))
        yb_test = np.load(os.path.join(path,test, 'labels.npy'))
        yc_test = np.load(os.path.join(path,test, 'rnr_labels.npy'))
    
        ids_testA = np.load(os.path.join(path, test, 'tweet_ids.npy'))
        ids_testBC = np.load(os.path.join(path, test, 'ids.npy'))    
        
        model = training(params, x_train, ya_train, yb_train, yc_train)
   
        pred_probabilities_a, pred_probabilities_b, pred_probabilities_c = model.predict(x_test,
                                                                                         verbose=0)    
        Y_pred_a = np.argmax(pred_probabilities_a, axis=2)
        Y_pred_b = np.argmax(pred_probabilities_b, axis=1)
        Y_pred_c = np.argmax(pred_probabilities_c, axis=1)
        
        maskB = np.any(yb_test, axis=1)
        ids_testB = list(compress(ids_testBC, maskB))
        yb_test = list(compress(yb_test, maskB))
        Y_pred_b = list(compress(Y_pred_b, maskB))
    
        ids_testB = np.asarray(ids_testB)
        yb_test = np.asarray(yb_test)
        Y_pred_b = np.asarray(Y_pred_b)
        yb_test = np.argmax(yb_test, axis=1)
        trees, tree_prediction, tree_label = branch2treelabels(ids_testB,
                                                               yb_test,
                                                               Y_pred_b) 
        treesC, tree_predictionC, tree_labelC = branch2treelabels(ids_testBC,
                                                                  yc_test,
                                                                  Y_pred_c)

        maskA = np.any(np.any(ya_test,axis=2),axis=1)
        
        if np.any(maskA):
        
            Y_true_a = ya_test[maskA]
            Y_true_a = np.argmax(Y_true_a, axis=2)
            Y_pred_a = Y_pred_a[maskA]
            ids_testA = ids_testA[maskA]
           
            fids_test = ids_testA.flatten()
            fy_pred = Y_pred_a.flatten()
            fy_test = Y_true_a.flatten()
            
            uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
            uniqtwid = uniqtwid.tolist()
            uindices2 = uindices2.tolist()
            del uindices2[uniqtwid.index(b'a')]
            del uniqtwid[uniqtwid.index(b'a')]
           
            uniq_dev_prediction = [fy_pred[i] for i in uindices2]
            uniq_dev_label =  [fy_test[i] for i in uindices2]
            
        else:
           
            uniq_dev_prediction = []
            uniq_dev_label = []
            uniqtwid = []
        
        perfold_result = {  
                      
                          'Task A':{'ID': uniqtwid,
                                    'Label':uniq_dev_label,
                                    'Prediction': uniq_dev_prediction
                                                },
                          'Task B':{'ID': trees,
                                    'Label':tree_label,
                                    'Prediction': tree_prediction},
                          'Task C':{'ID': treesC,
                                    'Label':tree_labelC,
                                    'Prediction': tree_predictionC}
                         }    
                          
        cv_ids_c.extend(treesC)
        cv_prediction_c.extend(tree_predictionC)
        cv_label_c.extend(tree_labelC)   
        
        cv_ids_b.extend(trees)
        cv_prediction_b.extend(tree_prediction)
        cv_label_b.extend(tree_label)   

        cv_ids_a.extend(uniqtwid)
        cv_prediction_a.extend(uniq_dev_prediction)
        cv_label_a.extend(uniq_dev_label)   
        
        allfolds.append(perfold_result) 
    
    Cmactest_P, Cmactest_R, Cmactest_F, _ = precision_recall_fscore_support(
                                                cv_label_c,
                                                cv_prediction_c,
                                                average='binary')    
    Cmictest_P, Cmictest_R, Cmictest_F, _ = precision_recall_fscore_support(
                                                cv_label_c,
                                                cv_prediction_c,
                                                average='binary')    
    Ctest_P, Ctest_R, Ctest_F, _ = precision_recall_fscore_support(
                                        cv_label_c,
                                        cv_prediction_c)    
    Cacc = accuracy_score(cv_label_c, cv_prediction_c)
      
        
    Bmactest_P, Bmactest_R, Bmactest_F, _ = precision_recall_fscore_support(
                                                cv_label_b,
                                                cv_prediction_b,
                                                labels=[0,1,2],
                                                average='macro')    
    Bmictest_P, Bmictest_R, Bmictest_F, _ = precision_recall_fscore_support(
                                                cv_label_b,
                                                cv_prediction_b,
                                                labels=[0,1,2],
                                                average='micro')    
    Btest_P, Btest_R, Btest_F, _ = precision_recall_fscore_support(
                                        cv_label_b, 
                                        cv_prediction_b,
                                        labels=[0,1,2])    
    Bacc = accuracy_score(cv_label_b, cv_prediction_b)
    
    
    Amactest_P, Amactest_R, Amactest_F, _ = precision_recall_fscore_support(
                                                cv_label_a,
                                                cv_prediction_a,
                                                labels=[0,1,2,3],
                                                average='macro')    
    Amictest_P, Amictest_R, Amictest_F, _ = precision_recall_fscore_support(
                                                cv_label_a,
                                                cv_prediction_a,
                                                labels=[0,1,2,3],
                                                average='micro')    
    Atest_P, Atest_R, Atest_F, _ = precision_recall_fscore_support(
                                        cv_label_a,
                                        cv_prediction_a,
                                        labels=[0,1,2,3])    
    Aacc = accuracy_score(cv_label_a, cv_prediction_a)
    
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
              'TaskC':{
                      'accuracy': Cacc,
                      'Macro': {'Macro_Precision': Cmactest_P,
                                'Macro_Recall': Cmactest_R,
                                'Macro_F_score': Cmactest_F},
                      'Micro': {'Micro_Precision': Cmictest_P,
                                'Micro_Recall': Cmictest_R,
                                'Micro_F_score': Cmictest_F}, 
                      'Per_class': {'Pclass_Precision': Ctest_P,
                                    'Pclass_Recall': Ctest_R,
                                    'Pclass_F_score': Ctest_F}
                      },
              'attachments': {'Task A':{'ID': cv_ids_a,
                                        'Label': cv_label_a,
                                        'Prediction': cv_prediction_a
                                        },
                              'Task B': {'ID': cv_ids_b,
                                         'Label': cv_label_b,
                                         'Prediction': cv_prediction_b 
                                        
                                      },
                              'Task C': {'ID': cv_ids_c,
                                         'Label': cv_label_c,
                                         'Prediction': cv_prediction_c
                                      },
                              'allfolds': allfolds
                       }
                } 
                              
    directory = "output"
    if not os.path.exists(directory):
        os.mkdir(directory)
        
    with open('output/output'+fname+'.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)
        
    return output


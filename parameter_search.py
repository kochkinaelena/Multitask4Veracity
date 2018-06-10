#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:23:32 2017

@author: Helen
"""
import pickle
import os
from hyperopt import fmin, tpe, hp, Trials 


def parameter_search(ntrials, objective_function, fname):

    
    search_space= { 'num_dense_layers': hp.choice('nlayers', [1,2]),
                    'num_dense_units': hp.choice('num_dense', [300, 400,
                                                               500, 600]), 
                    'num_epochs': hp.choice('num_epochs', [50]),
                    'num_lstm_units': hp.choice('num_lstm_units', [100, 200,
                                                                    300]),
                    'num_lstm_layers': hp.choice('num_lstm_layers', [1,2]),
                    'learn_rate': hp.choice('learn_rate', [1e-4, 1e-3]), 
                    'batchsize': hp.choice('batchsize', [32]),
                    'l2reg': hp.choice('l2reg', [ 1e-3])
                 
    }
    
    trials = Trials()
    
    best = fmin(objective_function,
        space=search_space,
        algo=tpe.suggest,
        max_evals=ntrials,
        trials=trials)
    
    params = trials.best_trial['result']['Params']
    
    directory = "output"
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    f = open('output/trials_'+fname+'.txt', "wb")
    pickle.dump(trials, f)
    f.close()
    
    filename = 'output/bestparams_'+fname+'.txt'
    f = open(filename, "wb")
    pickle.dump(params, f)
    f.close()
    
    return params

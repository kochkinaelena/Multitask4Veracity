#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:43:14 2017

@author: Helen
"""
import numpy as np

def branch2treelabels(ids_test,y_test, y_pred):
    
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_label = []
    
    for  tree in trees:
        treeindx = np.where(ids_test == tree)[0]
        tree_label.append(y_test[treeindx[0]])
        temp_prediction = [ y_pred[i] for i in treeindx] 
        # all different predictions from branches from one tree
        
        unique, counts = np.unique(temp_prediction,return_counts=True)
        
        tree_prediction.append(unique[np.argmax(counts)])
        
        
    return trees, tree_prediction, tree_label
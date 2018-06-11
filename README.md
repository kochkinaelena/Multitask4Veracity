# Multitask4Veracity
This repository contains code for the paper "All-in-one: Multi-task Learning for Rumour Stance classification,Detection and Verification" by E. Kochkina, M. Liakata, A. Zubiaga 

This code relies on preprocessed data that can be downloaded at https://figshare.com/articles/PHEME_dataset_Preprocessed_for_Multitask_Learning_for_Rumour_Verification/6473873

Raw data can be downloaded at https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078


## To run the code:

python outer.py

will be equivalent to running:

python outer.py --model='mtl2stance' --data='RumEval' --search=True --ntrials=10 --params="output/bestparams.txt" 

## outer.py has the following options:
--model - which task to train, stance or veracity
--data - which dataset to use 
--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false
-h, --help - explains the command line 


Required libraries:

 - Python 3
 - Keras
 - Hyperopt
 - Optparse
 

# Multitask4Veracity
This repository contains code for the paper "All-in-one: Multi-task Learning for Rumour Stance classification,Detection and Verification" by E. Kochkina, M. Liakata, A. Zubiaga 

This code relies on preprocessed data that can be downloaded at https://figshare.com/articles/PHEME_dataset_Preprocessed_for_Multitask_Learning_for_Rumour_Verification/6473873

Raw data can be downloaded at https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078


## How to run the code:

### Install prerequisites 

 * Python 3
 * Keras
 * Hyperopt
 * Optparse
 
### Run outer.py

#### outer.py has the following options:
* --model - which task to train, stance or veracity
* --data - which dataset to use 
* --search  - boolean, controls whether parameter search should be performed
* --ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
* --params - specifies filepath to file with parameters if --search is false
* -h, --help - explains the command line 

running

```
python outer.py
```

will be equivalent to running:
```
python outer.py --model='mtl2stance' --data='RumEval' --search=True --ntrials=10 --params="output/bestparams.txt" 
```

#### MTL2 Veracity + Stance

##### RumEval
```
python outer.py --model='mtl2stance' --data='RumEval' --search=True --ntrials=50
```
##### FullPHEME

5 folds
```
python outer.py --model='mtl2stance' --data='PHEME5' --search=True --ntrials=50
```
or

9 folds
```
python outer.py --model='mtl2stance' --data='PHEME9' --search=True --ntrials=50
```


#### MTL2 Veracity + Detection

5 folds
```
python outer.py --model='mtl2detect' --data='PHEME5' --search=True --ntrials=50
```
or

9 folds
```
python outer.py --model='mtl2detect' --data='PHEME9' --search=True --ntrials=50
```


#### MTL3 Veracity + Stance + Detection

5 folds
```
python outer.py --model='mtl3' --data='PHEME5' --search=True --ntrials=50
```

or

9 folds
```
python outer.py --model='mtl3' --data='PHEME9' --search=True --ntrials=50
```











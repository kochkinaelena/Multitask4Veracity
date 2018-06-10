"""
Run outer.py
python outer.py

outer.py has the following options:
python outer.py --model='mtl2stance' --data='RumEval' --search=True --ntrials=10 --params="output/bestparams.txt" 

--model - which task to train, stance or veracity
--data - which dataset to use 
--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false
-h, --help - explains the command line 

If performing parameter search, then execution will take long time 
depending on number of trials, size and number of layers in parameter space. 
Use of GPU is highly recommended. 
If running with default parametes then search won't be performed 
"""
import pickle
# os.environ["THEANO_FLAGS"]="floatX=float32"
# if using theano then use flag to set floatX=float32
from optparse import OptionParser
from parameter_search import parameter_search

from MTL2_RumEval_VeracityStance import eval_MTL2_RumEval
from MTL2_RumEval_VeracityStance import objective_MTL2_RumEval
from MTL2_CV_VeracityStance import eval_MTL2_stance_CV
from MTL2_CV_VeracityStance import objective_MTL2_stance_CV5
from MTL2_CV_VeracityStance import objective_MTL2_stance_CV9
from MTL2_CV_VeracityDetection import eval_MTL2_detection_CV
from MTL2_CV_VeracityDetection import objective_MTL2_detection_CV5
from MTL2_CV_VeracityDetection import objective_MTL2_detection_CV9
from MTL3_CV_VeracityStanceDetection import eval_MTL3
from MTL3_CV_VeracityStanceDetection import objective_MTL3_CV5
from MTL3_CV_VeracityStanceDetection import objective_MTL3_CV9
#%%


def main():
    parser = OptionParser()
    parser.add_option(
            '--search', dest='psearch', default=False,
            help='Whether parameter search should be done: default=%default')
    parser.add_option('--ntrials', dest='ntrials', default=10,
                      help='Number of trials: default=%default')
    parser.add_option(
            '--model', dest='model', default='mtl2stance',
            help='Which model to use. Can be one of the following:  mtl2stance, mtl2detect, mtl3; default=%default')
    parser.add_option(
            '--data', dest='data', default='RumEval',
            help='Which dataset to use: RumEval(train, dev, test) or PHEME5 or PHEME9 (leave one event out cross-validation): default=%default')
    (options, args) = parser.parse_args()
    psearch = options.psearch
    ntrials = int(options.ntrials)
    data = options.data
    model = options.model
    output = []
    if model == 'mtl2stance':
        if data == 'RumEval':
            if psearch:
                params = parameter_search(ntrials, objective_MTL2_RumEval,
                                          'MTL2_RumEval')
            else:
                params_file = 'bestparams_MTL2_RumEval.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_RumEval(params, 'MTL2_RumEval')
        elif data == 'PHEME5':
            if psearch:
                params = parameter_search(ntrials, objective_MTL2_stance_CV5,
                                          'MTL2_stance_PHEME5')
            else:
                params_file = 'bestparams_MTL2_stance_PHEME5.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_stance_CV(params,'PHEME5',
                                         'MTL2_stance_PHEME5')
        elif data == 'PHEME9':
            if psearch:
                params = parameter_search(ntrials, objective_MTL2_stance_CV9,
                                          'MTL2_stance_PHEME9')
            else:
                params_file = 'bestparams_MTL2_stance_PHEME9.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_stance_CV(params,'PHEME9',
                                         'MTL2_stance_PHEME9')
        else:
            print ("Check dataset name")
        
    elif model == 'mtl2detect':
        if data == 'PHEME5':
            if psearch:
                params = parameter_search(ntrials,
                                          objective_MTL2_detection_CV5,
                                          'MTL2_detection_PHEME5')
            else:
                params_file = 'bestparams_MTL2_detection_PHEME5.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_detection_CV(params,'PHEME5',
                                            'MTL2_detection_PHEME5')
        elif data == 'PHEME9':
            if psearch:
                params = parameter_search(ntrials,
                                          objective_MTL2_detection_CV9,
                                          'MTL2_detection_PHEME9')
            else:
                params_file = 'bestparams_MTL2_detection_PHEME9.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL2_detection_CV(params,'PHEME9',
                                         'MTL2_detection_PHEME9')
        else:
            print ("Check dataset name")
    elif model == 'mtl3':
        if data == 'PHEME5':
            if psearch:
                params = parameter_search(ntrials, objective_MTL3_CV5,
                                          'MTL3_PHEME5')
            else:
                params_file = 'bestparams_MTL3_PHEME5.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL3(params, 'PHEME5', 'MTL3_PHEME5')
        elif data == 'PHEME9':
            if psearch:
                params = parameter_search(ntrials,
                                          objective_MTL3_CV9,
                                          'MTL3_PHEME9')
            else:
                params_file = 'bestparams_MTL3_PHEME9.txt'
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
            print(params)
            output = eval_MTL3(params, 'PHEME9', 'MTL3_PHEME9')
        else:
            print ("Check dataset name")

    else:
       print ('Check model name') 
    return output
#%%

if __name__ == '__main__':
    output = main() 
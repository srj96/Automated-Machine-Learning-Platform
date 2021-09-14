from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn import utils
from random import seed
from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
from xgboost import XGBRegressor
import numpy as np 
import pickle 

class HyperOptimize:
    def __init__(self,param_grid,h_best = None):
        self.param_grid = param_grid
        self.h_best = h_best
    

    def objective_hopt(self,param):
        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
        m_name = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt", 'rb'))
        param = self.param_grid
        score = cross_val_score(m_name,f_file,l_file).mean()
        print("Display for HyperOpt : {}".format(score))
        return(score)
    
    def hyperopt(self,max_evals):
        space = {}
        for key,value in self.param_grid.items():
            first_param = sorted(value)[0]
            last_param = sorted(value)[-1]  
            j = {key : hp.quniform(key ,first_param,last_param,0.1)}
            space.update(j)
        self.h_best = fmin(fn=self.objective_hopt,space=space,algo=tpe.suggest,max_evals=max_evals)
        print("The best parameters->XGBoost by Hyper Opt {}".format(self.h_best))
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn import utils
from random import seed
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
import numpy as np 
import pickle

class BayesOptimize:
    def __init__(self,param_grid,b_best = None):
        
        super().__init__()
        self.param_grid = param_grid
        self.b_best = b_best
        
    def objective_bayes(self,**param):
        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
        m_name = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt", 'rb')) 
        param = self.param_grid
        score = cross_val_score(m_name, f_file, l_file).mean()
        print("Display for Bayes Opt : {}".format(score))
        return(score)
    
    def bayesoptimize(self,init_points,n_iter):
        pbounds = {}
        for key,value in self.param_grid.items():
            first_param = sorted(value)[0]
            last_param = sorted(value)[-1]  
            j = {key : (first_param,last_param)}
            pbounds.update(j)
        b_optimizer = BayesianOptimization(f=self.objective_bayes,pbounds=pbounds,random_state=1)
        b_optimizer.maximize(init_points = init_points , n_iter = n_iter)
        self.b_best = b_optimizer.max
        print("The best parameters->XGBoost by Bayes Opt {}".format(self.b_best)) 
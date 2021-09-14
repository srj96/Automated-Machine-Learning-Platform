import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn import utils
from random import seed
from sklearn.model_selection import GridSearchCV
import pickle

class GridSearch:
    def __init__(self,param_grid,g_best = None):
        super().__init__()
        self.param_grid = param_grid
        self.g_best = g_best  

    def grid_search_cv(self,cv):
        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
        m_name = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt", 'rb')) 
        grid_search = GridSearchCV(estimator = m_name, param_grid = self.param_grid, n_jobs=-1, cv= cv, verbose=2)
        gxb = grid_search.fit(f_file,l_file)
        self.g_best = gxb.best_params_
        print("The best parameters->XGBoost by grid search:{}".format(self.g_best))     
    

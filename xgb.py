from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import pickle 


class XgbRegression:
    def __init__ (self,label_predicted,model,n_estimators = None,max_depth = None,
                  colsample_bytree = None, learning_rate = None, gamma = None):
        
        self.label_predicted = label_predicted
        self.model = model 
        self.n_est = n_estimators
        self.max_depth = max_depth
        self.lr_rate = learning_rate
        self.colsample = colsample_bytree
        self.gamma = gamma 

    
    def xgbreg(self):
        np.random.seed(1)
        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
        f_test = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_test.txt",'rb'))
        self.model = XGBRegressor(objective = 'reg:squarederror',n_estimators= self.n_est,learning_rate=self.lr_rate,colsample_bytree=self.colsample,max_depth=self.max_depth,gamma=self.gamma)
        self.model.fit(f_file, l_file)
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt"
        pickle.dump(self.model,open(model_name,'wb'))
        self.label_predicted = self.model.predict(f_test)
        label_predicted_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_predicted.txt"
        pickle.dump(self.label_predicted,open(label_predicted_file,'wb'))
        return(self.label_predicted)    


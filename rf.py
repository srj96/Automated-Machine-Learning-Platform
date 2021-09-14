from sklearn.ensemble import RandomForestRegressor
from data_preprocess_reg import DataPreprocessing
import pandas as pd 
import numpy as np 
import pickle

class CogniRf :
    
    ''' This class contains the random forest model both regression and classification. 
    Same here the model as well as predicted labels are saved using pickle. '''
    
    def __init__(self, label_predicted,feature_test):
            
            super().__init__()
            
            self.label_predicted = label_predicted
            self.feature_test = feature_test
        
    # Random Forest Regression (trained model is saved) 
    def rf_reg(self):
        DataPreprocessing.split_train_test(self)
        np.random.seed(1)

        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(f_file,l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_rf.txt"

        pickle.dump(model, open(model_name,'wb'))

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    


    

        

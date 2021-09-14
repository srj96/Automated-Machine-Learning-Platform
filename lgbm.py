from lightgbm import LGBMRegressor
from data_preprocess_reg import DataPreprocessing
import pandas as pd
import numpy as np
import pickle 

class CogniLgb :

    '''This class contains the Light GBM model wherein both regression and classification
    are executed. Here also the model is saved with the predicted labels using pickle '''
    
    def __init__(self,label_predicted,feature_test):

                 super().__init__()
                 
                 self.label_predicted = label_predicted
                 self.feature_test = feature_test

    # Light GBM regression code (trained model is saved)
    def lgbreg(self):
        DataPreprocessing.split_train_test(self)
        np.random.seed(1)
        f_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt",'rb'))
        l_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt", 'rb'))
    
        model = LGBMRegressor()
        model.fit(f_file,l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_lgb.txt"
        # model_name = "model_lgb.txt"
        pickle.dump(model,open(model_name,'wb'))

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    

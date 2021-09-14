import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import pickle 

class MetricsCal:
    def __init__(self,l_predicted_file,l_test_file):

        self.l_predicted_file = l_predicted_file
        self.l_test_file = l_test_file 

        
    def metrics_score(self):
        
        self.l_predicted_file = pickle.load(open('/home/congnitensor/Python_projects/model_class/file_logs/label_predicted.txt', 'rb'))
        self.l_test_file = pickle.load(open("/home/congnitensor/Python_projects/model_class/file_logs/label_test.txt",'rb'))

        rmse = np.sqrt(mean_squared_error(self.l_test_file,self.l_predicted_file))
        error = abs(self.l_predicted_file - self.l_test_file)
        mape = np.mean(100 * (error / self.l_test_file))
        accuracy_mape = 100 - mape
        r_score = r2_score(self.l_test_file,self.l_predicted_file)
        print(" RMSE: %f" % (rmse), "\n" , "Accuracy by Mape : %f" %(accuracy_mape) ,"\n", "R squared score : %f" %(r_score))







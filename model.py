from data_preprocess import DataPreprocessing
from xgb import XgbRegression
from metrics import MetricsCal
import argparse
import os
import glob
class ModelWrapper(DataPreprocessing,XgbRegression,MetricsCal):
    def __init__(self,path,target,split_percent,n_estimators,max_depth,colsample_bytree,
                 learning_rate,gamma):

        super().__init__() 

        self.path = path
        self.target = target
        self.split_percent = split_percent

        self.colsample = colsample_bytree
        self.n_est = n_estimators
        self.gamma = gamma
        self.lr_rate = learning_rate
        self.max_depth = max_depth 
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--split_percent", help = "option for split percent", nargs = '?', const = 0.3 , type = float, default = 0.3)
    parser.add_argument("-t", "--target", help = "option for target value")
    parser.add_argument("-p", "--path_name", help = "enter the file path")
    parser.add_argument('-n_estimators',help = 'number of estimators',nargs = '?',const = 100,type = int,default = 100)
    parser.add_argument('-max_depth', help = 'maximum depth of trees',nargs = '?',const=5,type = int, default = 5)
    parser.add_argument('-learning_rate', help = 'learning rate of the model', nargs = '?', const = 0.3, type = float, default = 0.3)
    parser.add_argument('-colsample_bytree', help = 'fraction of the sample tree', nargs = '?', const = 0.0 , type = float , default = 0.0)
    parser.add_argument('-gamma', help = 'gamma parameter', nargs = '?', const = 0.0 , type = float, default = 0.0)
    parser.add_argument("-o","--option", help = "enter the option for load_data/description/split_data/xgb_model_run/reg_metrics")

    result = parser.parse_args()

    # print(result)

    obj = ModelWrapper(path = result.path_name, target = result.target, split_percent = result.split_percent,
                      n_estimators = result.n_estimators,max_depth = result.max_depth,colsample_bytree = result.colsample_bytree,
                      learning_rate = result.learning_rate, gamma = result.gamma)



    if result.option == "load_data":

        obj.load_data()

    if result.option == "description":

        obj.data_describe()

    if result.option == "split_data":

        obj.split_train_test()

    if result.option == "xgb_model_run":
        
        obj.xgbreg()

    
    if result.option == "reg_metrics":
        # not necessary as the files are already being save, 
        # as the code is dependent on previous steps for execution
        # if the user skips any previous steps then following may not 
        # execute unless the below functions are called again
        # obj.load_data()
        # obj.split_train_test()
        # obj.xgbreg()

        obj.metrics_score() 

    if result.option == "clean_up":
        
        log_file = '/home/congnitensor/Python_projects/model_class/file_logs/*'
        r = glob.glob(log_file)
        for ir in r:
            os.remove(ir)
         
        



##/home/congnitensor/Python_projects/model_class/BostonHousing.csv
## https://public-assets-ct.s3.us-east-2.amazonaws.com/testing/BostonHousing.csv


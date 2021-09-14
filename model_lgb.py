from data_preprocess import DataPreprocessing
from lgbm import CogniLgb
from metrics import MetricsCal
import argparse
import os
import glob

class LgbModelWrapper(DataPreprocessing,CogniLgb,MetricsCal):

    ''' Model base class Light GBM with arguments passed as split percent (-s)
    with target value (-t) being user defined.The path for the file being -p. 
    The parameters for xgboost are -max_depth, -n_estimators, -learning_rate,
    -num_leaves, -min_child_weight '''
    
    def __init__(self,**kwargs):
            
        super().__init__()

        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()] 

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--split_percent", help = "option for split percent", nargs = '?', const = 0.3 , type = float, default = 0.3)
    parser.add_argument("-t", "--target", help = "option for target value")
    parser.add_argument("-p", "--path_name", help = "enter the file path")
    parser.add_argument("-r", "--table_name", help = "enter the table name")
    parser.add_argument('-n_estimators',help = 'number of estimators',nargs = '?',const = 100,type = int,default = 100)
    parser.add_argument('-max_depth', help = 'maximum depth of trees',nargs = '?',const=-1,type = int, default = -1)
    parser.add_argument('-learning_rate', help = 'learning rate of the model', nargs = '?', const = 0.1, type = float, default = 0.1)
    parser.add_argument('-num_leaves', help = 'number of leaves', nargs = '?', const = 31 , type = int , default = 31)
    parser.add_argument('-min_child_weight', help = 'weight given to the child leaf', nargs = '?', const = 1.0 , type = float, default = 1e-3)
    parser.add_argument("-o","--option", help = "enter the option for load_data/description/split_data/lgb_reg_model_run/reg_metrics_lgb")

    result = parser.parse_args()

    obj = LgbModelWrapper(path = result.path_name, target = result.target,
                           table_name = result.table_name,split_percent = result.split_percent, 
                           num_leaves = result.num_leaves,
                           max_depth = result.max_depth, learning_rate = result.learning_rate,
                           min_child_weight = result.min_child_weight, n_estimators = result.n_estimators)
                        
    if result.option == "load_data":

        obj.load_data()

    if result.option == "description":

        obj.data_describe()

    if result.option == "split_data":

        obj.split_train_test()

    if result.option == "lgb_reg_model_run":
        
        obj.lgbreg()

    
    if result.option == "reg_metrics_lgbm":
        
        obj.metrics_cal_lgbmr()


    if result.option == "clean_up":
        
        log_file = '/home/congnitensor/Python_projects/model_class/file_logs/*'
        r = glob.glob(log_file)
        for ir in r:
            os.remove(ir)
         
        



##/home/congnitensor/Python_projects/model_class/BostonHousing.csv
## https://public-assets-ct.s3.us-east-2.amazonaws.com/testing/BostonHousing.csv


from data_preprocess import DataPreprocessing
from rf import CogniRf
from metrics import MetricsCal
import glob
import os
import argparse
import pickle

class RfModelWrapper(DataPreprocessing,CogniRf,MetricsCal):

    '''Model base class for Random Forest with arguments passed as split percent (-s)
    with target value (-t) being user defined.The path for the file being -p. 
    The parameters for xgboost are -max_depth, -n_estimators, -max_features,
    -min_samples_leaf, -min_samples_split'''

    def __init__(self,**kwargs):

        super().__init__()

        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()] 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--split_percent", help = "option for split percent", nargs = '?', const = 0.3 , type = float, default = 0.3)
    parser.add_argument("-t", "--target", help = "option for target value")
    parser.add_argument("-p", "--path_name", help = "enter the file path")
    parser.add_argument("-r", "--table_name", help = "enter the table name")
    parser.add_argument('-n_estimators',help = 'number of estimators',nargs = '?',const = 10,type = int,default = 10)
    parser.add_argument('-max_depth', help = 'maximum depth of trees',nargs = '?',const=0,type = int, default = 1)
    parser.add_argument('-max_features', help = 'maximum features', nargs = '?', const = 0, type = int, default = 10)
    parser.add_argument('-min_samples_leaf', help = 'minimum number of leaves', nargs = '?', const = 1 , type = int , default = 1)
    parser.add_argument('-min_samples_split', help = 'minimum of split sample', nargs = '?', const = 1.0 , type = float, default = 1.0)
    parser.add_argument("-o","--option", help = "enter the option for load_data/description/split_data/rf_reg_model_run/reg_metrics_rf")

    result = parser.parse_args()

    obj = RfModelWrapper(path = result.path_name, target = result.target, 
                         table_name = result.table_name, split_percent = result.split_percent,
                         n_est = result.n_estimators,
                         max_depth = result.max_depth,
                         min_samples_split = result.min_samples_split,
                         min_samples_leaf = result.min_samples_leaf, 
                         max_features = result.max_features)

    
    if result.option == "load_data":

        obj.load_data()

    if result.option == "description":

        obj.data_describe()

    if result.option == "split_data":

        obj.split_train_test()

    if result.option == "rf_reg_model_run":
        
        obj.rf_reg()

    
    if result.option == "reg_metrics_rf":

        
        obj.metrics_cal_rfr()


    if result.option == "clean_up":
        
        log_file = '/home/congnitensor/Python_projects/model_class/file_logs/*'
        r = glob.glob(log_file)
        for ir in r:
            os.remove(ir)
         
        



##/home/congnitensor/Python_projects/model_class/BostonHousing.csv
## https://public-assets-ct.s3.us-east-2.amazonaws.com/testing/BostonHousing.csv







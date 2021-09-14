from data_preprocess import DataPreprocessing
from regressions_models import RegClass
from metrics import MetricsCal
import argparse
import os
import glob
import json 
import ast

class RegressionBase(DataPreprocessing,RegClass,MetricsCal):

    '''Inherits Data preprocessing class, Regression Class and Metrics class and
    takes input as Json dictionary in main().
    '''
    
    def __init__(self,**kwargs):

        super().__init__()

        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()]
    

    def Reg_metrics_xgb(self):
        
        MetricsCal.metrics_cal_xgbr(self)
    
    def Reg_metrics_lgb(self):

        MetricsCal.metrics_cal_lgbmr(self)
    
    def Reg_metrics_rf(self):

        MetricsCal.metrics_cal_rfr(self)
    
    def Reg_metrics_adb(self):

        MetricsCal.metrics_cal_adbr(self)
    
    def Reg_metrics_elm(self):

        MetricsCal.metrics_cal_elmr(self)
    
    def Reg_metrics_svm(self):

        MetricsCal.metrics_cal_svmr(self)
    
    def Reg_metrics_knn(self):

        MetricsCal.metrics_cal_knnr(self)
    
    def Reg_metrics_dt(self):

        MetricsCal.metrics_cal_dtr(self)
    
    def Reg_metrics_sgd(self):

        MetricsCal.metrics_cal_sgdr(self)
    
    def Reg_metrics_lasso(self):

        MetricsCal.metrics_cal_lasso(self)
    
    def Reg_metrics_ridge(self):

        MetricsCal.metrics_cal_ridge(self)
    
    def Reg_metrics_mlp(self):

        MetricsCal.metrics_cal_mlpr(self)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("-s","--split_percent", help = "option for split percent", nargs = '?', const = 0.3 , type = float, default = 0.3)
    # parser.add_argument("-t", "--target", help = "option for target value")
    # parser.add_argument("-p", "--path_name", help = "enter the file path")
    # parser.add_argument("-r", "--table_name", help = "enter the table name")
    parser.add_argument("-j","--config_dict",help = 'enter the json dictionary',type = json.loads)
    parser.add_argument("-o","--option", help = "enter the option for regress_metrics_xgb/lgb/rf/elm/adb/svm/knn/dt/sgd/lasso/ridge/mlp/clean_up")

    result = parser.parse_args()
    dict_file = result.config_dict
    # with open(json_file,'r') as infile:
    #     dict_file = json.load(infile)

    # dict_file = ast.literal_eval(dict_file) 
    
    obj = RegressionBase(path = dict_file['path'], target = dict_file['target'],
                         split_percent= dict_file['split'], 
                         table_name = dict_file['table_name']) 
        
    # obj = RegressionBase(path = result.path_name, target = result.target,
    #                      split_percent= result.split_percent, 
    #                      table_name = result.table_name)

    if result.option == "regress_metrics_xgb":
        
        obj.Reg_metrics_xgb()
    
    if result.option == "regress_metrics_lgb":
        
        obj.Reg_metrics_lgb()
    
    if result.option == "regress_metrics_rf":

        obj.Reg_metrics_rf() 
    
    if result.option == "regress_metrics_svm":

        obj.Reg_metrics_svm()
    
    if result.option == "regress_metrics_adb":

        obj.Reg_metrics_adb()
    
    if result.option == "regress_metrics_elm":

        obj.Reg_metrics_elm()
    
    if result.option == "regress_metrics_knn":

        obj.Reg_metrics_knn()
    
    if result.option == "regress_metrics_dt":

        obj.Reg_metrics_dt()
    
    if result.option == "regress_metrics_sgd":

        obj.Reg_metrics_sgd()
    
    if result.option == "regress_metrics_lasso":
        
        obj.Reg_metrics_lasso()
    
    if result.option == "regress_metrics_ridge":

        obj.Reg_metrics_ridge()
    
    if result.option == "regress_metrics_mlp":

        obj.Reg_metrics_mlp()
        
    if result.option == "clean_up":
        
        log_file = '/home/congnitensor/Python_projects/model_class/file_logs/*'
        r = glob.glob(log_file)
        for ir in r:
            os.remove(ir)

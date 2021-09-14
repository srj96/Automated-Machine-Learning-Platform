from data_preprocess import DataPreprocessing
from classification_models import DTClassifier
from metrics_class import MetricsClass
import argparse
import json 
import os
import glob

class ClassificationBase(DataPreprocessing,DTClassifier,MetricsClass):

    '''Inherits Data preprocessing class, Classification Model Class and Metrics class and
    takes input as Json dictionary in main().
    '''
    
    def __init__(self,**kwargs):

        super().__init__()
        
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()]

    def class_metrics_xgb(self):

        MetricsClass.metrics_cal_xgbc(self)
    
    def class_metrics_lgb(self):

        MetricsClass.metrics_cal_lgbmc(self)
    
    def class_metrics_rf(self):

        MetricsClass.metrics_cal_rfc(self)
    
    def class_metrics_svm(self):

        MetricsClass.metrics_cal_svmc(self) 
    
    def class_metrics_adb(self):

        MetricsClass.metrics_cal_adbc(self)
    
    def class_metrics_elm(self):

        MetricsClass.metrics_cal_elmc(self)
    
    def class_metrics_knn(self):

        MetricsClass.metrics_cal_knnc(self)
    
    def class_metrics_dt(self):

        MetricsClass.metrics_cal_dtc(self)
    
    def class_metrics_nb(self):

        MetricsClass.metrics_cal_nbc(self)
    
    def class_metrics_sgd(self):

        MetricsClass.metrics_cal_sgdc(self) 
    
    def class_metrics_mlp(self):

        MetricsClass.metrics_cal_mlpc(self) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("-s","--split_percent", help = "option for split percent", nargs = '?', const = 0.3 , type = float, default = 0.3)
    # parser.add_argument("-t", "--target", help = "option for target value")
    # parser.add_argument("-p", "--path_name", help = "enter the file path")
    # parser.add_argument("-r", "--table_name", help = "enter the table name")
    parser.add_argument("-j","--config_dict",help = 'enter the json dictionary',type = json.loads)
    parser.add_argument("-o","--option", help = "enter the option for class_metrics_xgb/lgb/rf/adb/elm/svm/knn/dt/nb/sgd/mlp/clean_up")

    result = parser.parse_args()
    dict_file = result.config_dict

    obj = ClassificationBase(path = dict_file['path'], target = dict_file['target'],
                             split_percent= dict_file['split'],
                             table_name = dict_file['table_name']) 

    # obj = ClassificationBase(path = result.path_name, target = result.target,
    #                          split_percent= result.split_percent, 
    #                          table_name = result.table_name) 

    if result.option == "class_metrics_xgb":
        
        obj.class_metrics_xgb()
    
    if result.option == "class_metrics_lgb":
        
        obj.class_metrics_lgb()
    
    if result.option == "class_metrics_rf":

        obj.class_metrics_rf() 
    
    if result.option == "class_metrics_svm":

        obj.class_metrics_svm()
    
    if result.option == "class_metrics_adb":

        obj.class_metrics_adb()
    
    if result.option == "class_metrics_elm":

        obj.class_metrics_elm()
    
    if result.option == "class_metrics_knn":

        obj.class_metrics_knn()
    
    if result.option == "class_metrics_dt":

        obj.class_metrics_dt()
    
    if result.option == "class_metrics_nb":

        obj.class_metrics_nb()
    
    if result.option == "class_metrics_sgd":

        obj.class_metrics_sgd() 
    
    if result.option == "class_metrics_mlp":

        obj.class_metrics_mlp() 

    if result.option == "clean_up":
        
        log_file = '/home/congnitensor/Python_projects/model_class/file_logs/*'
        r = glob.glob(log_file)
        for ir in r:
            os.remove(ir)



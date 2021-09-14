import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
import requests 
from urllib.parse import urlparse
import io 
import os
import pickle 

class DataPreprocessing:
    '''This class executes various methods for data transformation for 
    regression and classification data with min-max scaling, normalization,
    categorical encoding-one-hot and label encoding.'''
    
    def __init__ (self,data = None,path = None,query = None, 
                  username = None, password = None, host = None , database = None,
                  target = None, split_percent = None,feature_train = None , feature_test = None , label_train = None , label_test = None):


                  self.data = data
                  self.path = path
                  self.query = query
                  self.username = username
                  self.password = password
                  self.host = host
                  self.database = database
                  self.target = target
                  self.split = split_percent 
                  self.feature_train = feature_train
                  self.feature_test = feature_test
                  self.label_train = label_train
                  self.label_test = label_test 

    
    def load_data(self):
        if os.path.splitext(self.path)[1] == ".csv":
            self.data = pd.read_csv(self.path)
        elif urlparse(self.path).scheme == 'https' and os.path.splitext(urlparse(self.path).path)[1] == ".csv":
            url = self.path
            s=requests.get(url).content
            self.data = pd.read_csv(io.StringIO(s.decode('utf-8')))
        elif os.path.splitext(self.path)[1] == ".db":
            db_string = "postgres+psycopg2://{}:{}@{}/{}".format(self.username , self.password , self.host , self.database)
            engine = create_engine(db_string)
            conn = engine.connect()
            self.data = pd.read_sql(self.query , con = conn)
        elif os.path.splittext(self.path)[1] == ".xlsx":
            self.data = pd.read_excel(self.path)
        else :
            print("--other file type entered--")

    
    def data_describe(self):
        self.load_data()
        print("Datatype :"'\n'"{}".format(self.data.dtypes))
    
    def split_train_test(self):
        self.load_data()
        label = self.data[self.target].values
        feature_col = [i for i in self.data.columns if i not in [self.target]]
        feature = self.data[feature_col]
        self.feature_train,self.feature_test,self.label_train,self.label_test = train_test_split(feature, label, test_size = self.split,random_state = 123)
        feature_train_file = "/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        label_train_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        feature_test_file = "/home/congnitensor/Python_projects/model_class/file_logs/feature_test.txt"
        label_test_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_test.txt"
        pickle.dump(self.feature_train,open(feature_train_file,'wb'))
        pickle.dump(self.label_train,open(label_train_file,'wb'))
        pickle.dump(self.feature_test,open(feature_test_file,'wb'))
        pickle.dump(self.label_test,open(label_test_file,'wb'))
        return(self.feature_train,self.feature_test,self.label_train,self.label_test)







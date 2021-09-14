from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from data_preprocess import DataPreprocessing
import pandas as pd 
import numpy as np 
import pickle
import warnings
warnings.filterwarnings("ignore") 

class RegClass:
    
    '''This class contains all the models relating to Regression. 
       Attributes : predicted labels(to be calculated), test features(from train-test split),
                    train feature file and train label file unpickled.'''

    def __init__(self,label_predicted,feature_test,f_file,l_file):   

        super().__init__()

        self.label_predicted = label_predicted
        self.feature_test = feature_test  

    # Opens the file containing the training set for feature and labels(target)

    def open_file(self):
        p_f = "/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        # p_f = "feature_train.txt"
        p_l = "/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        # p_l = "label_train.txt"
        with open(p_f,'rb') as feature_file:
            self.f_file= pickle.load(feature_file)
        with open(p_l,'rb') as label_file:
            self.l_file = pickle.load(label_file)
       
        return(self.f_file,self.l_file)
     
     
    '''Every function for the respective model calls DataPreprocessing class method
    data validation and open file method which unpickles trained feature and label file
     '''

    # XGBoost Regression model 

    def xgbreg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)
        
        model = XGBRegressor(objective = 'reg:squarederror')
        model.fit(self.f_file, self.l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file)
        
        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)    

    # Light GBM regression model

    def lgbreg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = LGBMRegressor()
        model.fit(self.f_file,self.l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_lgb.txt"
        # model_name = "model.txt"
        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file) 

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    
    # Random Forest regression model
    
    def rf_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = RandomForestRegressor(n_estimators=100)
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_rf.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    
    # Ada Boost regression model
    
    def adbreg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = AdaBoostRegressor()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_adb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    
    # Extreme Learning Machines regression model

    def elmreg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = ELMRegressor()
        stdsc = StandardScaler()

        train_feature_data = stdsc.fit_transform(self.f_file)
        test_feature_data = stdsc.fit_transform(self.feature_test)

        model.fit(train_feature_data,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_elm.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)
        
        self.label_predicted = model.predict(test_feature_data)
        return(self.label_predicted)
    
    # SVM regression model 
    
    def svm_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = svm.SVR(gamma='scale')
        feature_data = np.reshape(self.f_file,(len(self.f_file),-1))
        label_data = np.reshape(self.l_file,(len(self.l_file),-1))
        test_data = np.reshape(self.feature_test,(len(self.feature_test),-1))
        model.fit(feature_data,label_data)   

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_svm.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)
        
        self.label_predicted = model.predict(test_data) 
        return(self.label_predicted)  
    
    # K-Nearest Neighbour Regression Model

    def knn_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = KNeighborsRegressor()
        model.fit(self.f_file,self.l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_knn.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Decision Tree Regression

    def dt_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = DecisionTreeRegressor()

        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_dt.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Lasso Regression 

    def lasso_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)
        
        model = linear_model.Lasso()

        model.fit(self.f_file, self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_lasso.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)

    # Ridge Regression

    def ridge_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)
        
        model = linear_model.Ridge()

        model.fit(self.f_file, self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_ridge.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    
    # Stochastic Gradient Descent Regression 

    def sgd_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = SGDRegressor()

        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_sgd.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)
    
    # Multilayer Perceptron regression

    def mlp_reg(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = MLPRegressor()

        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_mlp.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)














    







    


    
    


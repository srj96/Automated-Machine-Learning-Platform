import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from data_preprocess import DataPreprocessing
import pickle
import warnings

warnings.filterwarnings("ignore")

class DTClassifier:

    '''This class contains all the models relating to Regression. 
       Attributes : predicted labels(to be calculated), test features(from train-test split),
                    train feature file and train label file unpickled.'''

    def __init__(self,label_predicted,feature_test):

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
     
    # Light GBM classifier model 

    def lgbmclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = LGBMClassifier()
        model.fit(self.f_file, self.l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_lgb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file) 

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)

    # XGBoost classifier model

    def xgbclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = XGBClassifier()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file)
        
        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 

    # Random Forest classifer model

    def rfclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = RandomForestClassifier()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_rf.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file)
        
        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Ada Boost classifier model
    
    def adbclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = AdaBoostClassifier()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_adb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file)
        
        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Extreme Learning Machines regression model

    def elmclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = ELMClassifier()
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
    
    # SVM classifier model
    
    def svmclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = svm.SVC(gamma = 'scale')
        feature_data = np.reshape(self.f_file,(len(self.f_file),-1))
        label_data = np.reshape(self.l_file,(len(self.l_file),-1))
        test_data = np.reshape(self.feature_test,(len(self.feature_test),-1))
        model.fit(feature_data,label_data) 

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_svm.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model,model_file)
        
        self.label_predicted = model.predict(test_data)
        return(self.label_predicted)
    
    # K-Nearest Neighbour Classification Model

    def knnclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = KNeighborsClassifier()
        model.fit(self.f_file,self.l_file)
        
        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_knn.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Decision Tree Classification

    def dtclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = DecisionTreeClassifier()

        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_dt.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 

    # Naive Bayes Classifier 

    def nbclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = GaussianNB()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_nb.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted) 
    
    # Stochastic Gradient Descent Classifier 

    def sgdclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = SGDClassifier()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_sgd.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)  
    
    # Multilayer Perceptron classifier

    def mlpclassifier(self):
        DataPreprocessing.data_validation(self)
        self.open_file()
        np.random.seed(1)

        model = MLPClassifier()
        model.fit(self.f_file,self.l_file)

        model_name = "/home/congnitensor/Python_projects/model_class/file_logs/model_mlp.txt"
        # model_name = "model.txt"

        with open(model_name,'wb') as model_file:
            pickle.dump(model, model_file)

        self.label_predicted = model.predict(self.feature_test)
        return(self.label_predicted)  














        






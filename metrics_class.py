import pandas as pd 
import numpy as np 
from classification_models import DTClassifier
from data_preprocess import DataPreprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

class MetricsClass:
    
    ''' This class contains the metrics calculation score for classification models only.
    This class contains the metrics calculation score for regression models only.
    They are namely F_1 score, Precision, Recall, Accuracy Score. 
    --- Note : Accuracy score = (TP + TN)/(TP + TN + FP + FN) 
    // Fraction of samples predicted correctly// ---
    Attributes : predicted labels from the model and test labels from the split.'''

    def __init__(self,label_predicted,label_test):
        
        super().__init__()

        self.label_predicted = label_predicted
        self.label_test = label_test

    # Metrics calculation for classification method namely f1 score, recall, precision, auc

    def metrics_score_class(self,l_predicted,l_test):
        f_one_score = np.mean(f1_score(l_test,l_predicted, average = 'macro'))
        recall = np.mean(recall_score(l_test,l_predicted, average = 'macro'))
        precision = precision_score(l_test,l_predicted,average= 'macro')
        accuracy = accuracy_score(l_test,l_predicted)
        print(" F-1 Score : %f" % (f_one_score),"\n","Recall Score : %f" % (recall),
              "\n","Precision Score : %f" % (precision), "\n" , 
              "Accuracy Score : %f" % (accuracy))
    

    '''Every metrics method for all the models call the model first to generate the predicted
    labels and then a metrics score method with arguments : predicted labels and test labels 
    '''

    # Calling metrics function for XGB classifier

    def metrics_cal_xgbc(self):

        DTClassifier.xgbclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)

    # Calling metrics function for Random Forest classifier

    def metrics_cal_rfc(self):

        DTClassifier.rfclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)
    
    # Calling metrics function for light GBM classifier
    
    def metrics_cal_lgbmc(self):

        DTClassifier.lgbmclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)
    
    # Calling metrics function for SVM classifier
    
    def metrics_cal_svmc(self):

        DTClassifier.svmclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)
    
    # Calling metrics function for Ada Boost classifier

    def metrics_cal_adbc(self):

        DTClassifier.adbclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test) 

    # Calling metrics function for ELM classifier

    def metrics_cal_elmc(self):

        DTClassifier.elmclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test) 
    
    
    # Calling metrics function for KNN classifier 

    def metrics_cal_knnc(self):

        DTClassifier.knnclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)
    
    # Calling metrics function for Decision Tree classifier 

    def metrics_cal_dtc(self):

        DTClassifier.dtclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)

    # Calling metrics function for Naive Bayes classifier

    def metrics_cal_nbc(self):

        DTClassifier.nbclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)
    
    
    # Calling metrics function for SGD classifier

    def metrics_cal_sgdc(self):

        DTClassifier.sgdclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)

    
    # Calling metrics function for MLP classifier

    def metrics_cal_mlpc(self):

        DTClassifier.mlpclassifier(self)

        self.metrics_score_class(self.label_predicted,self.label_test)

     




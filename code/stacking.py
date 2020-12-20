import numpy as np
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes

class Stacking():

    def __init__(self, models, meta_classifier, X_test):
        self.models = models
        self.meta_classifier = meta_classifier
        self.X_test = X_test

    def fit(self, X, y):
        N, D = X.shape
        
        models = self.models
        
        meta_features = np.zeros((N, len(models)))
        for i in range(len(models)):
            m = models[i]
            m.fit(X.copy(), y.copy())
            y_pred = m.predict(X.copy())
            
            meta_features[:, i] = y_pred
                
        self.meta_classifier.fit(meta_features, y.copy(), None)

    def predict(self, X):
        T, D = X.shape
        models = self.models
        
        meta_features = np.zeros((T, len(models)))
        for i in range(len(models)):
            m = models[i]
            y_pred = m.predict(X)
            
            meta_features[:, i] = y_pred
                
        return self.meta_classifier.predict(meta_features)

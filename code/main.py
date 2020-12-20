# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import utils
from random_forest import RandomForest, DecisionTree
from knn import KNN
from naive_bayes import NaiveBayes
from stacking import Stacking


def load_dataset(filename):
    with open(os.path.join('.','data',filename), 'rb') as f:
        return pd.read_csv(f)

def dataSetup():
    filename_train = "wordvec_train.csv"
    dataset_train = load_dataset(filename_train)
    X, y = dataset_train.iloc[:, :-1].to_numpy(), np.squeeze(dataset_train.iloc[:, -1:].to_numpy().astype(int))
    
    filename_test = "wordvec_test.csv"
    dataset_test = load_dataset(filename_test)
    X_test, y_test = dataset_test.iloc[:, :-1].to_numpy(), np.squeeze(dataset_test.iloc[:, -1:].to_numpy().astype(int))
    
    return X, y, X_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question
    start=datetime.now()

    if question == "1":
        # RANDOM FOREST
        X, y, X_test, y_test = dataSetup()
        print("Train and Test data loaded.")
        
        model = RandomForest(num_trees=15,k=6)
        utils.evaluate_model(model, X, y, X_test, y_test)
        
    elif question == "2":
        # Naive Bayes
        X, y, X_test, y_test = dataSetup()
        print("Train and Test data loaded.")
        
        model = NaiveBayes()
        utils.evaluate_model(model, X, y, X_test, y_test)

    elif question == "3":
        # KNN
        X, y, X_test, y_test = dataSetup()
        print("Train and Test data loaded.")
        
        model = KNN(3)
        utils.evaluate_model(model, X, y, X_test, y_test)

    elif question == "4":
        # Stacking
        X, y, X_test, y_test = dataSetup()
        print("Train and Test data loaded.")
        
        model0 = NaiveBayes()
        model1 = KNN(3)
        model2 = RandomForest(num_trees=15,k=7)
        meta_model = DecisionTree(max_depth=np.inf)
        
        model = Stacking([model0, model1, model2], meta_model, X_test)
        utils.evaluate_model(model, X, y, X_test, y_test)

    else:
        print("Unknown question: %s" % question)
    
    print("Elapsed time: ", datetime.now()-start)


import matplotlib.pyplot as plt
import pandas as pd

from algos import *
from sklearn.preprocessing import scale, OrdinalEncoder, LabelEncoder

def load_data():
    data = pd.read_csv('car.csv')

    X,y = data.values[:, 0:6], data.values[:, 6]
    enc = OrdinalEncoder()
    enc.fit(X)
    X = enc.transform(X)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return X,y

def main():
    X,y = load_data()
    #visualize_data_2D(X,y, 'Car')
    #visualize_data_3D(X,y, 'Car')
    #kmeans(X,y, 'Car')
    #expectation_maximization(X,y, 'Car')
    #pca(X,y,'Car')
    #ica(X,y, 'Car')
    #randomized_projection(X,y, 'Car')
    #elect_k_best(X,y, 'Car')
    reduce_then_cluster(X,y, 'Car')

if __name__ == "__main__":
    main()


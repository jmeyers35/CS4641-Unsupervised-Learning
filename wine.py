import matplotlib.pyplot as plt
import pandas as pd

from algos import *
from sklearn.preprocessing import scale

def load_data():
    data = pd.read_csv('red-wine.csv')
    X,y = data.values[:, 0:11], data.values[:, 11]
    return scale(X), y


def main():
    X,y = load_data()
    #visualize_data_2D(X,y, 'Wine')
    #visualize_data_3D(X,y, 'Wine')
    #kmeans(X,y, 'Wine')
    #expectation_maximization(X,y, 'Wine')
    #pca(X,y, 'Wine')
    #ica(X,y, 'Wine')
    #randomized_projection(X,y, 'Wine')
    #select_k_best(X,y, 'Wine')
    #reduce_then_cluster(X,y, 'Wine')
    #reduce_then_neural_net(X,y, 'Wine', 10)
    cluster_then_neural_net(X,y, 'Wine')


if __name__ == "__main__":
    main()
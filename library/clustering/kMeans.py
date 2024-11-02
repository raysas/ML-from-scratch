'''
K-means clustering algorithm

This module contains the implementation of the K-means clustering algorithm.  
The algorithm is implemented in the class kMeans, which is a subclass of the Clustering class.
'''

from datetime import datetime
import os
from IPython.display import display
from ..utils.helper import *
import numpy as np  
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt


class kMeans:
    ''' 
    ------------------------------------
        K-means clustering algorithm
    ------------------------------------
    
    This class instanciates a clustering model: the K-means algorithm to cluster the data into k clusters (by default, with random initializtion of means)  
    The distance function used here by defaultwill be the manhattan distance. It connectes to teh distances defined in utils.py
    Note that this is a non parametric clustering approach;
    it has a limitation taht the number of cluster has to be predefined, finding the right number of K is an open problem.  
    In practice we apply some heuristics to get an optimal number of clusters, one approach is the silhouette method implemented in metrics.py


    ### The idea behind this algorithm:

    1. randomly assign all points in random clusters, this is initially set by random initilization of means for each cluster, here by randomly choosing any k points as means 
    2. compute the distances between each data point and the means, each data point will be assigned to the cluster that it's closer to. 
    3. recompute teh means based on the new values in each cluster
    4. iterate from step 2 and repeat until convergence

    The temp distance matrix at each iteration will be stored in a matrix of shape (n, k) where n is the number of data points and k is the number of clusters of this form:

    |      | cluster1_mean | cluster2_mean | ... | clusterk_mean |  
    |------|---------------|---------------|-----|---------------|  
    |point1| distance1 | distance2 | ... | distancek |  
    |...   |...        |...        |...   |...|  
    |pointn| distance1 | distance2 | ... | distancek |  

    ### The class has the following attributes:  

    - data: the data to be clustered  
    - k: the number of clusters  
    - distance: the distance function to be used to compute the distances between the data points (or clusters) and the means  
    - max_iter: the maximum number of iterations to be run before stopping the algorithm  

    ### The class has the following methods:  

    - assign_means: assigns the means to the clusters, this is done randomly by default  
    - fit: fits the model to the data, this is the main method that runs the algorithm (implements it from parent class)

    ### Associations:  

    It associates with the class Cluster in a composition (has-a) relationship, it can not exist without it  


    To be further revised  


    '''
    
    def __init__(self, data, k, distance=cluster_distance, max_iter=100):
        self.__data = data
        self.__k = k
        self.__distance = distance
        self.__max_iter = max_iter
        self.__clusters = []

    def __str__(self):
        return f'K-means clustering model with {self.__k} clusters'
    def __repr__(self):
        return f'kMeans(data={self.__data}, k={self.__k}, distance={self.__distance}, max_iter={self.__max_iter})'

    def __setattr__(self, name: str, value: Any) -> None:
        if name == '_kMeans__data':
            if not isinstance(value, pd.DataFrame):
                raise ValueError('The data should be a pandas DataFrame')
            
        elif name == '_kMeans__k':
            if not isinstance(value, int):
                raise ValueError('The number of clusters should be an integer')
            
        elif name == '_kMeans__distance':
            # make sure this fucntion exists
            if not callable(value):
                raise ValueError('The distance function should be a callable')

        elif name == '_kMeans__max_iter':
            if not isinstance(value, int):
                raise ValueError('The maximum number of iterations should be an integer')

        self.__dict__[name] = value

    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, value):
        self.__data = value
    @data.deleter
    def data(self):
        del self.__data
    
    @property
    def k(self):
        return self.__k
    @k.setter
    def k(self, value):
        self.__k = value
    @k.deleter
    def k(self):
        del self.__k
    
    @property
    def distance(self):
        return self.__distance
    @distance.setter
    def distance(self, value):
        self.__distance = value
    @distance.deleter
    def distance(self):
        del self.__distance

    @property
    def max_iter(self):
        return self.__max_iter
    @max_iter.setter
    def max_iter(self, value):
        self.__max_iter = value
    @max_iter.deleter
    def max_iter(self):
        del self.__max_iter

    @property
    def clusters(self):
        return self.__clusters
    @clusters.setter
    def clusters(self, value):
        self.__clusters = value
    @clusters.deleter
    def clusters(self):
        del self.__clusters

    def assign_means(self, seed=42):
        '''
        Assigns the means to the clusters, this is done randomly by default
        '''
        means = []
        np.random.seed(seed)

        indices= np.random.choice(range(len(self.__data)), size=self.__k, replace=False) #avoids dups
        for _ in range(k):
            mu=[]
            for __ in range(self.__data.shape[1]):
                mu.append(float(self.__data[indices[_], __])) #fixes the freakinggggggggg format
            means.append(mu)
        return means
    

    def fit(self, verbose=True, cache=True, seed=42):

        cache_distance =[]
        cache_means = []

        ################################## 1. Assign means ##################################

        # -- assign k means randomly

        # assign_means(random=True)
        # M1=list(np.array(df.loc['Gitanes'],dtype=float))
        # M2=list(np.array(df.loc['Lucky Strike'],dtype=float))
        # means=[M1,M2]

        # --
        df=self.__data
        k=self.__k
        distance=self.__distance
        max_iter=self.__max_iter

        if verbose:
            print(f'Dataset:')
            display(df)

        means = self.assign_means(seed)

        for mu in means:
            for _ in range(len(mu)):
                mu[_] = float(mu[_])

        means=[list(mu) for mu in means]

        cache_means.append(means)
        if verbose:
            print(f'\t\t-------------- Initial means: --------------\n')
            display(pd.DataFrame(means, columns=[df.columns],index=[f'Cluster {_}' for _ in range(1,k+1)]))


        # -- loop prep
        iter_count=0; convergence=False

        while (not convergence and iter_count<max_iter):
            # ---- counting the iterations ----
            iter_count+=1
            if verbose:
                print(f'-------------------\nIteration: {iter_count}\n-------------------')
            # ---------------------------------

            ################################## 2. Assign each data point to a cluster ##################################
            ################################## 2.a Compute the distances between each data point and the means ##################################

            distances = np.zeros((n,k))
            
            for i in range(n):
                for j in range(k):
                    distances[i,j] = distance(data[i], means[j])
            distance_df = pd.DataFrame(distances, index=df.index, columns=[f'Cluster {i+1}' for i in range(k)])
            
            cache_distance.append(distance_df)
            if verbose:
                print('\t\t--- Distances between points and means ---')
                display(distance_df)

            ################################## 2.b Assign each data point to the cluster with the closest mean ##################################

            cluster_set=[Cluster() for _ in range(k)]
            index_cluster_set=[Cluster() for _ in range(k)]

            for i in range(n):
                min_index=np.argmin(distances[i])
                cluster_set[min_index]+=data[i]
                index_cluster_set[min_index]+=i

            ################################## 3. Compute the new means of the clusters ##################################

            means=[]
            for i in range(k):
                means.append(cluster_set[i].center)

            if verbose:
                print('\t\t\t--- Newly assigned means ---')
                display(pd.DataFrame(means, columns=[df.columns],index=[f'Cluster {_}' for _ in range(1,k+1)]))

            cache_means.append(means)

            # --------- checks if the means have changed -------

            if iter_count>1:
                convergence=[cache_means[-1][_]==cache_means[-2][_] for _ in range(k)]
                convergence=all(convergence)

            if verbose:
                print(f'Convergence state: {convergence}\n\n')

            ################################## 4. repeat until stopping criterion is met ##################################

        if cache:
            os.makedirs('__cache__', exist_ok=True)
            dir_name = f'__cache__/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            os.makedirs(dir_name, exist_ok=True)
                        
            # --- an excel file where each sheet is an item (pd df) of cache_distance
            with pd.ExcelWriter(f'{dir_name}/distance_matrices.xlsx') as writer:
                for i in range(len(cache_distance)):
                    cache_distance[i].to_excel(writer, sheet_name=f'Iteration_{i+1}')

            # --- a csv file where each row is an item in cache_means
            means_df = pd.DataFrame(cache_means, columns=[f'Cluster {i+1}' for i in range(k)], index=['Initialization']+[f'Iteration {i}' for i in range(1,len(cache_means))])
            means_df.to_csv(f'{dir_name}/means_progression.csv')

        return index_cluster_set
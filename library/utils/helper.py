import functools
from typing import Any
from collections.abc import Iterable
from statistics import mean


class Cluster:
    def __init__(self, cluster_list=[]):
        self.__cluster_list = cluster_list
        self.__center = self.computeCentroid()

    def __setattr__(self, name: str, value: Any) -> None:
        

        if name == '_Cluster__cluster_list':
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                value = [x for x in value]
            self.__dict__[name] = value
            self.__center = self.computeCentroid() # -- automatically reruns computeCentroid when cluster_list is updated

        else:
            self.__dict__[name] = value

    def __str__(self):
        return f'Cluster: {self.cluster_list};\n\tCentroid: {self.center}'
    def __repr__(self):
        return f'Cluster(cluster_list={self.cluster_list})'
    
    @property
    def cluster_list(self):
        return self.__cluster_list
    @cluster_list.setter
    def cluster_list(self, value):
        self.__cluster_list = value
    @cluster_list.deleter
    def cluster_list(self):
        del self.__cluster_list

    @property
    def center(self):
        return self.__center
    # @center.setter shouldnt be allowed

    

    def __iter__(self):
        return iter(self.cluster_list)

    def __add__(self, other):
        if type(other) != Cluster:
            other = Cluster([other])
        return Cluster(self.cluster_list + other.cluster_list)
    
    def __sub__(self, other):
        if type(other) != Cluster :
            other = Cluster([other])
            # print('other:', other) #debug
        return Cluster([x for x in self if x not in other]) # set difference
    
    def __len__(self):
        return len(self.cluster_list)
    
    def __getitem__(self, index):
        return self.cluster_list[index]
    

    def computeCentroid(self, algo='mean'):
        if len(self.__cluster_list) == 0:
            return None
        
        if len(self.__cluster_list) == 1: 
            return self.__cluster_list[0] #there's only one element in the cluster, so the centroid is the element itself
        # if each item is a primitive type, i wanna find the mean; if each is an nd array, i wanna return an nd array

        centroid = None
        for item in self.__cluster_list:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                transposed = zip(*self.__cluster_list)
                centroid = [mean(coordinate) for coordinate in transposed]
            else:
                centroid = float(mean(self.__cluster_list))
        return centroid
    

    
def validate_dimensions(func):
    @functools.wraps(func) # --->preserves teh function signature
    def wrapper(point1, point2, *args, **kwargs):
        # -- convert from scalar to vector (1D array)
        if isinstance(point1, (float, int)):
            point1 = [point1]
        if isinstance(point2, (float, int)):
            point2 = [point2]

        # -- check if the two points have the same number of dimensions (same domain)
        if len(point1) != len(point2):
            raise ValueError(f'The two points should have the same number of dimensions (as in # of cols); {len(point1)} != {len(point2)}')

        return func(point1, point2, *args, **kwargs)
    return wrapper

@validate_dimensions
def manhattan_distance(point1, point2):
    '''
    Computes and returns the Manhattan distance between two points x=(x1, x2... xm) and y=(y1, y2,... ym)
    '''

    # -- convert from scalar to vector (1D array)
    if isinstance(point1, float) or isinstance(point1, int):
        point1 = [point1]
    if isinstance(point2, float) or isinstance(point2, int):
        point2 = [point2]

    if len(point1) != len(point2):
        raise ValueError(f'The two points should have the same number of dimensions (as in # of cols); {len(point1)} != {len(point2)}')
    
    distance = sum(abs(point1[i] - point2[i]) for i in range(len(point1)))
    return float(round(distance,5))

def cluster_distance(c1,c2, func=manhattan_distance, method='centroid'):
    '''
    Computes the distance between two clusters c1 and c2 using the specified method.
    The method can be one of 'single', 'complete', 'average', 'centroid'

    In kmeans clustering, the method is by default centroid (force it in the function)
    '''

    c1=Cluster(c1)
    c2=Cluster(c2)
        
    print(f'c1: {c1}; \nc2: {c2}') #debug
    if method == 'single':
        return min(func(x1,x2) for x1 in c1 for x2 in c2)
    
    elif method == 'complete':
        return max(func(x1,x2) for x1 in c1 for x2 in c2)
    
    elif method == 'average':
        return sum(func(x1,x2) for x1 in c1 for x2 in c2) / (len(c1) * len(c2))
    
    elif method == 'centroid':
        centroid_1 = c1.center
        centroid_2 = c2.center

        return func(centroid_1, centroid_2)
     

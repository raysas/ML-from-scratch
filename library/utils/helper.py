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
    

    def computeCentroid(self):
        '''
        Computes the centroid of the cluster based on average of the coordinates of the points in the cluster  
        If the item in the cluster is a vector (iterable)  
            centroid = [mean(x1), mean(x2), ... mean(xm)] (maybe consider doing a tuple instead of a list)  
        If the item in the cluster is a scalar  
            centroid = mean(x1)  
        where x1, x2, ... xm are the coordinates of the points in the cluster (items in one item in the cluster)

        - If the cluster is empty, return None
        - If the cluster has only one item, return the item itself (float or list)
        - If the cluster has more than one item, return the centroid as described above (list or float)

        _e.g., if the cluster is [[1,2,3], [4,5,6], [7,8,9]]; that's x1, x2 and x3, it's gonna be [mean(1,4,7), mean(2,5,8), mean(3,6,9)] = [4,5,6]_  
        ```
        >>> c = Cluster([[1,2,3], [4,5,6], [7,8,9]])
        >>> c.computeCentroid()
        [4.0, 5.0, 6.0]
        ```  
        _if the cluster is [1,2,3]; that's only x1 per data point, it's gonna be mean(1,2,3) = 2_
        ```
        >>> c = Cluster([1,2,3])
        >>> c.computeCentroid()
        2.0
        ```
        _Some other examples:_
        ```
        >>> c = Cluster([(1,2)])
        >>> c.center
        [1.0, 2.0]
        >>> c = Cluster([np.array([1,2]), np.array([3,4])])
        >>> c.center
        [2.0, 3.0]
        >>> c = Cluster(np.array([1,2]))
        >>> c.center
        1.0
        '''
        if len(self.__cluster_list) == 0:
            return None
        
        if len(self.__cluster_list) == 1: 
            if isinstance(self.__cluster_list[0], Iterable) and not isinstance(self.__cluster_list[0], (str, bytes)):
                return list(self.__cluster_list[0])
            return float(self.__cluster_list[0]) #there's only one element in the cluster (1dim), so the centroid is the element itself

        centroid = None
        for item in self.__cluster_list:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                transposed = zip(*self.__cluster_list)
                centroid = [mean(coordinate) for coordinate in transposed]
            else:
                centroid = float(mean(self.__cluster_list))
        return centroid
    

    
def validate_dimensions(func):
    '''
    Decorator to validate the dimensions of the two points passed to the distance function
    '''
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
    formula: sum(abs(x1-y1) + abs(x2-y2) + ... + abs(xm-ym))
    '''
    distance = sum(abs(point1[i] - point2[i]) for i in range(len(point1)))
    return float(round(distance,5))

@validate_dimensions
def euclidean_distance(point1, point2):
    '''
    Computes and returns the Euclidean distance between two points x=(x1, x2... xm) and y=(y1, y2,... ym)

    formula: sqrt(sum((x1-y1)^2 + (x2-y2)^2 + ... + (xm-ym)^2))
    '''
    distance = sum((point1[i] - point2[i])**2 for i in range(len(point1))) ** 0.5
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
     


class Dataset:
    '''
    makes the dataset object unified for all types of data; what's important is:  
    -  iterable  
    - indexable  
    - has a name  
    - has colnames and rownames  
    - has a shape (rows, cols)  
    - has a target y vector (if it's a supervised learning dataset)  
    - display shows it in a tabular form 

    It can be deduced that the dataset is a tabular:  
    pd.DataFrame, np.ndarray, list of lists, list of tuples, list of dicts, dict of lists, dict of dicts, etc.  
    '''
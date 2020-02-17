import json
import math # contains sqrt, exp, pow, etc.
import random
import time
import random

from copy import deepcopy
from collections import deque


class TravelingSalesmanProblem:
    """ Representation of a traveling salesman optimization problem.
    
    An instance of this class represents a complete circuit of the cities
    in the `path` attribute.
    
    
    Parameters
    ----------
    cities : iterable
        An iterable sequence of cities; each element of the sequence must be
        a tuple (name, (x, y)) containing the name and coordinates of a city
        on a rectangular grid. e.g., ("Atlanta", (585.6, 376.8))
        
    shuffle : bool
        If True, then the order of the input cities (and therefore the starting
        city) is randomized.
    
    Attributes
    ----------
    names : sequence
        An iterable sequence (list by default) containing only the names from
        the cities in the order they appear in the current TSP path

    coords : sequence
        An iterable sequence (list by default) containing only the coordinates
        from the cities in the order they appear in the current TSP path

    path : tuple
        A path between cities as specified by the order of the city
        tuples in the list.
    """
    def __init__(self, cities, shuffle=False):
        if shuffle:
            cities = list(cities)
            random.shuffle(cities)
        self.path = tuple(cities)  # using a tuple makes the path sequence immutable
        self.__utility = None  # access this attribute through the .utility property
        
    def copy(self, shuffle=False):
        cities = list(self.path)
        if shuffle: random.shuffle(cities)
        return TravelingSalesmanProblem(cities)
    
    @property
    def names(self):
        """Strip and return only the city name from each element of the
        path list. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> ["Atlanta", ...]
        """
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        """ Strip the city name from each element of the path list and
        return a list of tuples containing only pairs of xy coordinates
        for the cities. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
        """
        _, coords = zip(*self.path)
        return coords
    
    @property
    def utility(self):
        """ Calculate and cache the total distance of the path in the
        current state.
        """
        if self.__utility is None:
            self.__utility = self.__get_value()
        return self.__utility
        
    def dist(self, xy1, xy2):
        """ Calculate the distance between two points.
        
        You may choose to use Euclidean distance, Manhattan distance, or some
        other metric
        """
        euclidean_distance = math.sqrt( (xy1[0]- xy2[0])**2 + (xy1[1]- xy2[1])**2 )
        return euclidean_distance

    def successors(self):
        """ Return a list of states in the neighborhood of the current state.
        
        You may define the neighborhood in many different ways; although some
        will perform better than others. One method that usually performs well
        for TSP is to generate neighbors of the current path by selecting a
        starting point and an ending point in the current path and reversing
        the order of the nodes between those boundaries.
        
        For example, if the current list of cities (i.e., the path) is [A, B, C, D]
        then the neighbors will include [B, A, C, D], [C, B, A, D], and [A, C, B, D].
        (The order of successors does not matter.) 
        
        Returns
        -------
        iterable<Problem>
            A list of TravelingSalesmanProblem instances initialized with their list
            of cities set to one of the neighboring permutations of cities in the
            present state
        """       
        
        for offset in range(len(self.path) - 1):
            # print('offset', offset)
            for width in range(2, len(self.path) - 1):
                # print('width' , width)
                nodes = deque(self.path)
                # print(nodes)
                nodes.rotate(offset)
                # print('after rotate', nodes)
                path = [nodes.popleft() for _ in range(width)][::-1] + list(nodes)
                # print('path', path)
                yield TravelingSalesmanProblem(path)
    
    def get_successor(self):
        """ Return a random state from the neighborhood of the current state.
        
        You may define the neighborhood in many different ways; although some
        will perform better than others. One method that usually performs well
        for TSP is to generate neighbors of the current path by selecting a
        starting point and an ending point in the current path and reversing
        the order of the nodes between those boundaries.
        
        For example, if the current list of cities (i.e., the path) is [A, B, C, D]
        then the neighbors will include [B, A, C, D], [C, B, A, D], and [A, C, B, D].
        (The order of successors does not matter.) 

        Returns
        -------
        list<Problem>
            A list of TravelingSalesmanProblem instances initialized with their list
            of cities set to one of the neighboring permutations of cities in the
            present state
        """
   
        offset = random.randint(0, len(self.path) - 2)
        width = random.randint(2, len(self.path) - 2)
        nodes = deque(self.path)
        nodes.rotate(offset)
        path = [nodes.popleft() for _ in range(width)][::-1] + list(nodes)
        return TravelingSalesmanProblem(path)

    def __get_value(self):
        """ Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of cities in the path
        sequence. 
        
        For example, if the current path is (A, B, C, D) then the total path length is:
            
            dist = DIST(A, B) + DIST(B, C) + DIST(C, D) + DIST(D, A)
        
        You may use any distance metric that obeys the triangle inequality (e.g.,
        Manhattan distance or Euclidean distance) for the DIST() function.
        
        Since the goal of our optimizers is to maximize the value of the objective
        function, multiply the total distance by -1 so that short path lengths
        are larger numbers than long path lengths. 
        
        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list
        
        Notes
        -----
            (1) Remember to include the edge from the last city back to the first city
            
            (2) Remember to multiply the path length by -1 so that short paths have
                higher value relative to long paths
        """
        
        total_path = 0
        for i in range(len(self.path)):
            if i == len(self.path) - 1:
                total_path +=  self.dist(self.path[0][1], self.path[i][1])
            else:
                total_path +=  self.dist(self.path[i][1], self.path[i+1][1])
        
        return round(total_path, 2) * -1    
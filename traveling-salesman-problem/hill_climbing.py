class HillClimbingSolver:
    """ 
    Parameters
    ----------
    epochs : int
        The upper limit on the number of rounds to perform hill climbing; the
        algorithm terminates and returns the best observed result when this
        iteration limit is exceeded.
    """
    
    def __init__(self, epochs=100):
        self.epochs = epochs
    
    def solve(self, problem):
        """ Optimize the input problem by applying greedy hill climbing.

        Parameters
        ----------
        problem : Problem
            An initialized instance of an optimization problem. The Problem class
            interface must implement a callable method "successors()" which returns
            a iterable sequence (i.e., a list or generator) of the states in the
            neighborhood of the current state, and a property "utility" which returns
            a fitness score for the state. (See the `TravelingSalesmanProblem` class
            for more details.)

        Returns
        -------
        Problem
            The resulting approximate solution state of the optimization problem
            
        
        Notes
        -----
            (1) DO NOT include the MAKE-NODE line from the AIMA pseudocode
        """
        
        for _ in range(self.epochs):
            neighbor = max(problem.successors(), key=lambda x: x.utility)
            if neighbor.utility < problem.utility: break
            problem = neighbor
        return problem
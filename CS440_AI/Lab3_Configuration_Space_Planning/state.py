import copy

from itertools import count
# NOTE: using this global index means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO: implement this method
def manhattan(a, b):
    """
    Computes the manhattan distance
    @param a: a length-3 state tuple (x, y, shape)
    @param b: a length-3 state tuple
    @return: the manhattan distance between a and b
    """
    return 0


from abc import ABC, abstractmethod
class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # __hash__ method allow us to keep track of which 
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass


# State: a length 3 list indicating the current location in the grid and the shape
# Goal: a tuple of locations in the grid that have not yet been reached
#   NOTE: it is more efficient to store this as a binary string...
# maze: a maze object (deals with checking collision with walls...)
# mst_cache: You will not use mst_cache for this MP. reference to a dictionary which caches a set of goal locations to their MST value
class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, mst_cache={}, use_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally, 
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.mst_cache = mst_cache # DO NOT USE
        self.maze_neighbors = maze.getNeighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # TODO: implement this method
    # Unlike MP 2, we do not need to remove goals, because we only want to reach one of the goals
    def get_neighbors(self, ispart1=False):
        nbr_states = []

        # We provide you with a method for getting a list of neighbors of a state
        # that uses the Maze's getNeighbors function.
        neighboring_locs = self.maze_neighbors(*self.state, part1=ispart1)

        return nbr_states

    # TODO: implement this method
    def is_goal(self):
        pass

    # TODO: implement these methods __hash__ AND __eq__
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return True

    # TODO: implement this method
    # Our heuristic is: manhattan(self.state, nearest_goal). No need for MST.
    def compute_heuristic(self):
        return 0
    
    # TODO: implement this method. It should be similar to MP 2
    def __lt__(self, other):
        pass
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)

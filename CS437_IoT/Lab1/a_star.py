class Node():
    def __init__(self,last_direction=None,position=None):
        self.last_direction = last_direction
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __eq__(self,target)-> bool:
        return self.position == target.position

    def __lt__(self,target)-> bool:
        return self.f < target.f
    
    def __gt__(self,target)-> bool:
        return self.f > target.f

def astar(maze,start,end):
    start_node = Node(None,)
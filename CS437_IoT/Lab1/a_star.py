class Node():
    def __init__(self,parent = None,last_direction=None,position=None):
        self.last_direction = last_direction
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __eq__(self,target)-> bool:
        return self.position == target.position

    def __lt__(self,target)-> bool:
        return self.f < target.f
            

def astar(maze,start,end):
    start_node = Node(None,None,start)
    end_node = Node(None,None,end)
    open_list = []
    closed_list = []
    open_list.append(start_node)
    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f <= current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            direction = []
            current = current_node
            while current.last_direction is not None:
                direction.append(current.last_direction)
                path.append(current.position)
                current = current.parent
            path.reverse()
            direction.reverse()
            return path,direction
        
        children = []
        for new_pos,direction in zip([(0,1),(0,-1),(-1,0),(1,0)],['r','l','f','b']):
            node_pos = (current_node.position[0] + new_pos[0], current_node.position[1] + new_pos[1])
            if node_pos[0] > (len(maze) - 1) or node_pos[0] < 0 or node_pos[1] > (len(maze[len(maze)-1]) -1) or node_pos[1] < 0:
                continue
            if maze[node_pos[0]][node_pos[1]] != 0:
                continue
            new_node = Node(current_node,direction,node_pos)
            children.append(new_node)
        
        for child in children:
            for closed_child in closed_list:
                if child == closed_child:
                    continue
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h
            for open_node in open_list:
                if child == open_node and child.g >= open_node.g:
                    continue
            open_list.append(child)
    
    
def main():

    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]

    start = (9, 9)
    end = (0, 0)

    path,direction = astar(maze, start, end)
    print(path,"\n",direction)


if __name__ == '__main__':
    main()
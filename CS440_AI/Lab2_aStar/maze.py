# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021, 
# Inspired by previous work by Michael Abir (abir2@illinois.edu) and Rahul Kunji (rahulsk2@illinois.edu)

from collections import namedtuple
from itertools import chain 

class MazeError(Exception):
    pass

class Maze:
    """
    creates a maze instance given a `path` to a file containing characters in `legend`. 
    """
    def __init__(self, path, legend = {'wall': '%', 'start': 'P', 'waypoint': '.'}):
        self.path = path
        # Passed in legend cannot introduce anything new
        for key in 'wall', 'start', 'waypoint':
            if key not in legend:
                raise ValueError('undefined legend key \'{0}\''.format(key))
        
        # Creates a legend to abstract away ASCII, e.g. self.legend.wall
        self.legend = namedtuple('legend', ('wall', 'start', 'waypoint'))(
            legend['wall'], 
            legend['start'], 
            legend['waypoint'])
        
        with open(path) as file:
            lines = tuple(line.strip() for line in file.readlines() if line)
        
        # Stores copy of ASCII maze in self._storage as well as dimensions in self.size.x/y
        n = len(lines)
        m = min(map(len, lines))
        
        if any(len(line) != m for line in lines):
            raise MazeError('(maze \'{0}\'): all maze rows must be the same length (shortest row has length {1})'.format(path, m))
        
        self._storage   = lines 
        self.size       = namedtuple('size', ('x', 'y'))(m, n)
        
        if any(self[x] != self.legend.wall for x in chain(
            ((    0, j) for j in range(m)), 
            ((n - 1, j) for j in range(m)), 
            ((i,     0) for i in range(n)), 
            ((i, m - 1) for i in range(n)))):
            raise MazeError('(maze \'{0}\'): maze borders must only contain `wall` cells (\'{1}\')'.format(path, self.legend.wall))
        if n < 3 or m < 3:
            raise MazeError('(maze \'{0}\'): maze dimensions ({1}, {2}) must be at least (3, 3)'.format(path, n, m))
        
        # Checks if only 1 start, if so, stores index in self.start
        self.start  = None 
        for x in ((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x) if self[i, j] == self.legend.start):
            if self.start is None:
                self.start = x
            elif type(self.start) is int:
                self.start += 1 
            else: 
                self.start  = 2
        if type(self.start) is int or self.start is None:
            raise MazeError('(maze \'{0}\'): maze must contain exactly one `start` cell (\'{1}\') (found {2})'.format(
                path, self.legend.start, 0 if self.start is None else self.start))
        
        # Stores waypoint indices in self.waypoints
        self.waypoints = tuple((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x) if self[i, j] == self.legend.waypoint)
        
        # there is no point in making this private since anyone trying to cheat 
        # could simply overwrite the underscored variable
        self.states_explored    = 0
    
    def __getitem__(self, index):
        """Access data at index via self[index] instead of using self._storage"""
        i, j = index
        if 0 <= i < self.size.y and 0 <= j < self.size.x:
            return self._storage[i][j]
        else:
            raise IndexError('cell index ({0}, {1}) out of range'.format(i, j))
    
    def indices(self):
        """Returns generator of all indices in maze"""
        return ((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x))
    
    def navigable(self, i, j):
        """Check if moving to (i,j) is a valid move"""
        try:
            return self[i, j] != self.legend.wall 
        except IndexError:
            return False 

    def neighbors(self, i, j):
        """Returns list of neighboing squares that can be moved to from the given row,col"""
        self.states_explored += 1 
        return tuple(x for x in (
            (i + 1, j),
            (i - 1, j),
            (i, j + 1),
            (i, j - 1)) 
            if self.navigable( * x ))

    def validate_path(self, path):
        # validate type and shape 
        if len(path) == 0:
            return 'path must not be empty'
        if not all(len(vertex) == 2 for vertex in path):
            return 'each path element must be a two-element sequence'
        
        # normalize path in case student used an element type that is not `tuple` 
        path = tuple(map(tuple, path))

        # check if path is contiguous
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if sum(abs(b - a) for a, b in zip(a, b)) != 1:
                return 'path vertex {1} ({4}, {5}) must be exactly one move away from path vertex {0} ({2}, {3})'.format(
                    i, i + 1, * a , * b )

        # check if path is navigable 
        for i, x in enumerate(path):
            if not self.navigable( * x ):
                return 'path vertex {0} ({1}, {2}) is not a navigable maze cell'.format(i, * x )
        
        # check if path ends at a waypoint 
        for waypoint in self.waypoints:
            if path[-1] == waypoint:
                break 
        else:
            return 'last path vertex {0} ({1}, {2}) must be a waypoint'.format(len(path) - 1, * path[-1] )

        # check for unnecessary path segments
        indices = {}
        for i, x in enumerate(path):
            if x in indices:
                if all(self[x] != self.legend.waypoint for x in path[indices[x] : i]):
                    return 'path segment [{0} : {1}] contains no waypoints'.format(indices[x], i)
            indices[x] = i 
        
        # check if path contains all waypoints 
        for i, x in enumerate(self.waypoints):
            if x not in indices:
                return 'waypoint {0} ({1}, {2}) was never visited'.format(i, * x )


# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021, 
# Inspired by previous work by Michael Abir (abir2@illinois.edu) and Rahul Kunji (rahulsk2@illinois.edu)

import argparse
import pygame

class gradient:
    def __init__(self, start, end):
        # rgb colors
        self.start  = start 
        self.end    = end 
    
    def __getitem__(self, fraction):
        t = fraction[0] / max(1, fraction[1] - 1) # prevent division by zero
        return tuple(max(0, min(start * (1 - t) + end * t, 255)) 
            for start, end in zip(self.start, self.end))

class agent:
    def __init__(self, position, maze):
        self.position   = position 
        self.maze       = maze 

    def move(self, move):
        position = tuple(i + move for i, move in zip(self.position, move))
        if self.maze.navigable( * position ):
            previous        = self.position
            self.position   = position 
            return previous,
        else: 
            return ()
            
class Application:
    def __init__(self, human = True, scale = 20, fps = 30, alt_color = False):
        self.running    = True
        self.scale      = scale
        self.fps        = fps
        
        self.human      = human 
        # accessibility for colorblind students 
        if alt_color:
            self.gradient = gradient((64, 224, 208), (139, 0, 139))
        else:
            self.gradient = gradient((255, 0, 0), (0, 255, 0))

    def run(self, maze, path=[], save=None):
        self.maze = maze
        
        self.window = tuple(x * self.scale for x in self.maze.size)

        if self.human:
            self.agent = agent(self.maze.start, self.maze)
            
            path            = []
            states_explored = 0
        else:
            states_explored = self.maze.states_explored
            
        pygame.init()
        
        self.surface = pygame.display.set_mode(self.window, pygame.HWSURFACE)
        self.surface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(maze.path)

        if self.human:
            self.draw_player()
        else:
            print("""
Results 
{{
    path length         : {0}
    states explored     : {1}
}}
            """.format(len(path), states_explored))
            
            self.draw_path(path)

        self.draw_maze()
        self.draw_start()
        self.draw_waypoints()

        pygame.display.flip()
        
        if type(save) is str:
            pygame.image.save(self.surface, save)
            self.running = False
        
        clock = pygame.time.Clock()
        
        while self.running:
            pygame.event.pump()
            clock.tick(self.fps)
            
            for event in pygame.event.get():
                if      event.type == pygame.QUIT:
                    raise SystemExit
                elif    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
                elif    event.type == pygame.KEYDOWN and self.human:
                    try:
                        move = {
                            pygame.K_RIGHT  : ( 0,  1),
                            pygame.K_LEFT   : ( 0, -1),
                            pygame.K_UP     : (-1,  0),
                            pygame.K_DOWN   : ( 1,  0),
                        }[event.key] 
                        path.extend(self.agent.move(move))
                    except KeyError: 
                        pass
                
                    self.loop(path + [self.agent.position])

    # The game loop is where everything is drawn to the context. Only called when a human is playing
    def loop(self, path):
        self.draw_path(path)
        self.draw_waypoints()
        self.draw_player()
        pygame.display.flip()

    # Draws the path (given as a list of (row, col) tuples) to the display context
    def draw_path(self, path):
        for i, x in enumerate(path):
            self.draw_square( * x , self.gradient[i, len(path)])
    
    # Draws the full maze to the display context
    def draw_maze(self):
        for x in self.maze.indices():
            if self.maze[x] == self.maze.legend.wall: 
                self.draw_square( * x )
    
    def draw_square(self, i, j, color = (0, 0, 0)):
        pygame.draw.rect(self.surface, color, tuple(i * self.scale for i in (j, i, 1, 1)), 0)
    
    def draw_circle(self, i, j, color = (0, 0, 0), radius = None):
        if radius is None:
            radius = self.scale / 4
        pygame.draw.circle(self.surface, color, tuple(int((i + 0.5) * self.scale) for i in (j, i)), int(radius))

    # Draws the player to the display context, and draws the path moved (only called if there is a human player)
    def draw_player(self):
        self.draw_circle( * self.agent.position , (0, 0, 255))

    # Draws the waypoints to the display context
    def draw_waypoints(self):
        for x in self.maze.waypoints:
            self.draw_circle( * x )

    # Draws start location of path
    def draw_start(self):
        i, j = self.maze.start
        pygame.draw.rect(self.surface, (0, 0, 255), tuple(int(i * self.scale) for i in (j + 0.25, i + 0.25, 0.5, 0.5)), 0)

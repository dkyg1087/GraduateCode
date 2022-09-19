from state import WordLadderState, EightPuzzleState, SingleGoalGridState, GridState
from search import best_first_search
from utils import read_puzzle, read_word_ladders
from maze import Application, Maze

import time
import argparse

def main(args):
    if args.problem_type == "WordLadder":
        word_ladder_problems = read_word_ladders()
    
        for start_word, goal_word in word_ladder_problems:
            print(f"Doing WordLadder from {start_word} to {goal_word}")
            start = time.time()
            starting_state = WordLadderState(start_word, goal_word, dist_from_start=0, use_heuristic=args.use_heuristic)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            print("\tPath found:", [p for p in path])
            print(f"\tTime: {end-start:.3f}")

    elif args.problem_type == "EightPuzzle":
        print(f"Doing EightPuzzle for length {args.puzzle_len} puzzles")
        all_puzzles = read_puzzle(f"data/eight_puzzle/{args.puzzle_len}_moves.txt")
        for puzzle in all_puzzles:
            start = time.time()
            start_puzzle = puzzle[0]
            zero_loc = puzzle[1]
            print(f"Start puzzle: {start_puzzle}")
            # import numpy as np
            # goal_puzzle = np.arange(9).reshape(3,3).tolist()
            goal_puzzle = [[0,1,2],[3,4,5],[6,7,8]]
            starting_state = EightPuzzleState(start_puzzle, goal_puzzle, 
                                dist_from_start=0, use_heuristic=args.use_heuristic, zero_loc=zero_loc)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            # print("\tPath found:", [p for p in path])
            print(f"\tTime: {end-start:.3f}")
        
    elif "Grid" in args.problem_type:
        # MAZE -------
        filename = args.maze_file
        print(f"Doing Maze search for file {filename}")
        maze = Maze(filename)
        path = []
        if not args.human:
            start = time.time()  
            if args.problem_type == "GridSingle":
                starting_state = SingleGoalGridState(maze.start, maze.waypoints, 
                                dist_from_start=0, use_heuristic=args.use_heuristic,
                                maze_neighbors=maze.neighbors)
            else:
                starting_state = GridState(maze.start, maze.waypoints, 
                                dist_from_start=0, use_heuristic=args.use_heuristic,
                                maze_neighbors=maze.neighbors, mst_cache={})
            path = best_first_search(starting_state)
            end = time.time()

            print("\tGoals: ", maze.waypoints)
            print("\tStart: ", maze.start)
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            print("\tStates explored: ", maze.states_explored)
            # print("Path found:", [p for p in path])
            print("\tTime:", end-start)
        
        if args.show_maze_vis or args.human:
            path = [s.state for s in path]
            application = Application(args.human, args.scale, args.fps, args.altcolor)
            application.run(maze, path, args.save_maze)
    else:
        print("Problem type must be one of [WordLadder, EightPuzzle, GridSingle, GridMulti]")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP2 Search')
    # WORDLADDER ARGS
    parser.add_argument('--problem_type',dest="problem_type", type=str,default="WordLadder",
                        help='Which search problem (i.e., State) to solve: [WordLadder, EightPuzzle, GridSingle, GridMulti]')
    
    # EIGHTPUZZLE ARGS
    parser.add_argument('--puzzle_len',dest="puzzle_len", type=int, default = 5,
                        help='EightPuzzle problem difficulty: one of [5, 10, 27]')
    
    # MAZE ARGS
    parser.add_argument('--maze_file', type=str, default="data/mazes/grid_single/tiny",
                        help = 'path to maze file')
    parser.add_argument('--show_maze_vis', default = False, action = 'store_true',
                        help = 'show maze visualization')
    parser.add_argument('--human', default = False, action = 'store_true',
                        help = 'run in human-playable mode')
    parser.add_argument('--use_heuristic', default = True, action = 'store_true',
                        help = 'use heuristic h in best_first_search')
    
    # You do not need to change these
    parser.add_argument('--scale',  dest = 'scale', type = int, default = 20,
                        help = 'display scale')
    parser.add_argument('--fps',    dest = 'fps', type = int, default = 30,
                        help = 'display framerate')
    parser.add_argument('--save_maze', dest = 'save_maze', type = str, default = None,
                        help = 'save output to image file')
    parser.add_argument('--altcolor', dest = 'altcolor', default = False, action = 'store_true',
                        help = 'view in an alternate color scheme')

    args = parser.parse_args()
    main(args)
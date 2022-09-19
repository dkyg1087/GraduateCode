
# EightPuzzle ------------------------------------------------------------------------------------------------

def read_puzzle(filename):
    with open(filename, "r") as file:
        all_grids = []
        for line in file:
            grid = [[]]
            for c in line.strip():
                if len(grid[-1])==3:
                    grid.append([])
                intc = int(c)
                if intc == 0:
                    zero_loc = [len(grid)-1, len(grid[-1])]
                grid[-1].append(intc)
            all_grids.append([grid, zero_loc])
        return all_grids

# WordLadder ------------------------------------------------------------------------------------------------
def read_word_ladders():
    with open("data/word_ladder/ladder_problems.txt", "r") as file:
        all_pairs = []
        for line in file:
            words = line.split()
            if len(words) > 0:
                all_pairs.append(words)
    return all_pairs
    
with open("data/word_ladder/wiki-100k.txt", "rb") as word_file:
    english_words = set(word.strip().decode("utf-8").lower() for word in word_file if word[0] !="#")

def is_english_word(word):
    return word.lower() in english_words

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# Maze ------------------------------------------------------------------------------------------------

# given a list/tuple of objectives and a distance function on those objectives
# return the weight of the Minimum Spanning Tree among those objectives
def compute_mst_cost(objectives, distance):
    mst = MST(objectives, distance)
    return mst.compute_mst_weight()

# TODO: this is not efficient because it forces us to recompute all the between goal distances each time we call MST...
class MST:
    def __init__(self, nodes, distance):
        self.elements = {key: None for key in nodes}
        self.distances   = {
                (i, j): distance(i, j)
                for i, j in self.cross(nodes)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)
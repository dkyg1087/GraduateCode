import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        # TODO: write your function here
        if self._train and (self.a is not None):
            reward = -0.1
            if dead:
                reward  = -1
            elif points > self.points:
                reward = 1
            
            self.N[self.s][self.a]+=1
            
            lr = self.C / (self.C + self.N[self.s][self.a])
            self.Q[self.s][self.a]+= lr * (reward + self.gamma * max(self.Q[s_prime])-self.Q[self.s][self.a])
                   
        if dead:
            self.reset()
            return utils.UP
        else:
            self.s = s_prime
            self.points = points
            
            for i,n in enumerate(np.flip(self.N[self.s])):
                if n < self.Ne:
                    action = len(self.actions) - i -1
                    self.a = action
                    return action
            
            action = len(self.actions) - np.argmax(np.flip(self.Q[s_prime]))-1 # make the priority RIGHT > LEFT > DOWN > UP.
            self.a = action
            return action

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment
        #* DONE
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        fdx,fdy,awx,awy,abt,abb,abl,abr = 0,0,0,0,0,0,0,0
        
        if snake_head_x < food_x : 
            fdx = 2
        elif snake_head_x > food_x:
            fdx = 1
        
        if snake_head_y < food_y :
            fdy = 2
        elif snake_head_y > food_y:
            fdy = 1
        
        height, width = utils.DISPLAY_HEIGHT, utils.DISPLAY_WIDTH
        if snake_head_x == 1:
            awx = 1
        elif snake_head_x == width-2:
            awx = 2
        
        if snake_head_y == 1:
            awy = 1
        elif snake_head_y == height-2:
            awy = 2
        
        for adj in [(1,0,"r"),(-1,0,"l"),(0,-1,"t"),(0,1,"b")]:
            if (snake_head_x + adj[0],snake_head_y + adj[1]) in snake_body:
                if adj[2] == "r":
                    abr = 1
                elif adj[2] == "l":
                    abl = 1
                elif adj[2] == "t":
                    abt = 1
                else:
                    abb = 1
        
        return (fdx,fdy,awx,awy,abt,abb,abl,abr)

import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            a = random.randrange(self.action_size)
        else:
            ### CODE ####
            # Choose the best action
            state = torch.from_numpy(state)
            state = torch.unsqueeze(state, 0)
            state = state.to(device)
            Q_values = self.policy_net(state)
            index = torch.argmax(Q_values)
            a = index.item()
            
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)
        mask = mask.to(device)
        
        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        Q_values_actions_current = self.policy_net(states)
        indices = actions.unsqueeze(-1)
        Q_values_current = torch.gather(Q_values_actions_current, dim=1, index = indices)
        
        # Compute Q function of next state
        ### CODE ####
        next_states = torch.from_numpy(next_states).cuda()
        with torch.no_grad():
          Q_values_actions_next = self.policy_net(next_states)
        
        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
          
          Q_values_max = torch.max(Q_values_actions_next, 1)[0] * mask
        
        # Compute the Huber Loss
        ### CODE ####
        expected_state_action_values = rewards + (self.discount_factor * Q_values_max)
        criterion = nn.HuberLoss(reduction='mean')
        loss = criterion(expected_state_action_values, torch.squeeze(Q_values_current, 1))

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


import numpy as np
import utils

# For checkpoint1
# Parameters are:
# snake_head_x=5, snake_head_y=5, food_x=2, food_y=2, Ne=40, C=40, gamma=0.7
true_Q = utils.load('data/checkpoint1.npy')
true_N = utils.load('data/checkpoint1_N.npy')

# For checkpoint2
# Parameters are:
# snake_head_x=5, snake_head_y=5, food_x=2, food_y=2, Ne=20, C=60, gamma=0.5
# Uncomment the two lines below for checkpoint2
# true_Q = utils.load('data/checkpoint2.npy')
# true_N = utils.load('data/checkpoint2_N.npy')

# For checkpoint3
# Parameters are:
# snake_head_x=3, snake_head_y=3, food_x=10, food_y=4, Ne=30, C=30, gamma=0.6
# Uncomment the two lines below for checkpoint3
# true_Q = utils.load('data/checkpoint3.npy')
# true_N = utils.load('data/checkpoint3_N.npy')

Q = utils.load('checkpoint.npy')
N = utils.load('checkpoint_N.npy')

print(f'Your Q matrix is correct: {np.array_equal(true_Q, Q)}')
print(f'Your N matrix is correct: {np.array_equal(true_N, N)}')
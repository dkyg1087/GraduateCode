from matplotlib import pyplot as plt
import numpy as np

def print_2Dmaze(ax:plt.axes,maze:np.array):
    ax.cla()
    ax.pcolor(maze,edgecolors='k',linewidths=1)
    plt.pause(.05)
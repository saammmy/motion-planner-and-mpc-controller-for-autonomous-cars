
from matplotlib import pyplot as plt


def plot_trajectory(plot, t, x,y,v,a,j):
    
    fig, axs = plot
    [x.clear() for x in axs.flat]
    axs[0, 0].plot(x, y, 'tab:red')
    axs[0, 1].plot(t, v, 'tab:orange')
    axs[1, 0].plot(t, a, 'tab:green')
    axs[1, 1].plot(t, j, 'tab:blue')

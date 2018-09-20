#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:34:20 2018

@author: tommy

See:
    https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/
"""

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from KDEpy import FFTKDE

def main():
    here = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(here, r'_static/img/')
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    
    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))
    
    # Plot a scatter that persists (isn't redrawn) and the initial line.
    np.random.seed(123)
    distribution = stats.norm()
    data = distribution.rvs(128)
    ax.set_title('Kernel density estimation animated', fontsize=16)
    ax.scatter(data, np.zeros_like(data), color='red', marker='|', zorder=10,
               label='Data')
    
    
    #x = np.arange(0, 20, 0.1)
    #ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
    x = np.linspace(np.min(data) - 1, np.max(data) + 1, num=2**10)
    y = FFTKDE(bw=1.05**(0 - 84)).fit(data)(x)
    line, = ax.plot(x, y, linewidth=2, label='KDE')
    ax.plot(x, distribution.pdf(x), linewidth=2, label='True distribution')
    ax.grid(True, zorder=-55)
    
    plt.legend(fontsize=12)
    
    def update(i):
        label = 'timestep {0}'.format(i)
        
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        bw = 1.05**(i - 64)
        y = FFTKDE(bw=bw).fit(data)(x)
        line.set_ydata(y)
        bw_formatted = str(round(bw, 3)).ljust(5, '0')
        ax.set_xlabel('Bandwidth $h$: {}'.format(bw_formatted), fontsize=14)
        print(label, bw)
        return line, ax
    
    
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(1, 128), interval=25)
    
    # plt.show() will just loop the animation forever.
    anim.save(os.path.join(save_path, r'KDE_bw_animation.gif'), 
              dpi=80, writer='imagemagick')
    plt.show()
    
if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how a weighted KDE may be used to animate the distribution
of log returns for stocks. Requires pandas_datareader and pandas.
"""

import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import datetime

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DOWNLOAD = False

if DOWNLOAD:
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2019, 1, 1)
    web.DataReader("F", "iex", start, end).to_csv("stocks.csv")


data = pd.read_csv("stocks.csv")


def weight_function(arr):
    """
  Weight function for weighting samples backwards in time less.
  """
    k = 0.005
    return 0.045 * np.exp(-arr * k)


# Compute the log returns and the weights
stock_data = np.log(data.close.values[1:] / data.close.values[:-1])
weights = weight_function(np.arange(0, len(stock_data)))

# Iterate over the number of points in the plot, and create new plots
# This is not computatationally efficient, but it's reasonably fast
# The following UNIX command combines the images to a GIF
# $ convert -delay 10 -loop 0 kde*.png stocks_animation.gif
points = (list(range(1, 20)) + list(range(20, 100, 2)) + list(range(100, 300, 3))
+ list(range(300, len(stock_data), 5)))
for i, num_points in enumerate(points):

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [3, 1]}, sharey="row"
    )

    # Create a kernel density estimate, the bandwidth is found by trial and error
    x, y = (
        FFTKDE(bw=0.002)
        .fit(stock_data[:num_points], weights=weights[num_points - 1 :: -1])
        .evaluate()
    )

    # The left-most plot
    ax1.set_title("Stock data (IEX)")
    ax1.plot(stock_data[:num_points], label="daily log returns")
    ax1.plot(np.arange(num_points), weights[num_points - 1 :: -1], label="weights")
    ax1.set_ylim([-0.05, 0.05])
    ax1.grid(True, zorder=5, alpha=0.5, ls="--")
    ax1.legend(loc="upper left")

    # The right-most plot
    ax2.set_title("Time weighted KDE")
    ax2.plot(y, x, color=COLOR_CYCLE[2])
    ax2.fill_betweenx(x, 0, y, alpha=0.2, color=COLOR_CYCLE[2])

    # https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
    rgba_colors = np.zeros((num_points, 4))
    # for red the first column needs to be one
    rgba_colors[:, 0] = 31 / 256
    rgba_colors[:, 1] = 119 / 256
    rgba_colors[:, 2] = 180 / 256
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = (weights[num_points - 1 :: -1] / (0.045)) ** 10

    ax2.scatter(
        np.zeros_like(stock_data[:num_points]) + 1,
        stock_data[:num_points],
        color=rgba_colors,
        marker="x",
        s=25,
    )
    ax2.set_xlim([0, 50])
    ax2.grid(True, zorder=5, alpha=0.5, ls="--")
    ax2.get_xaxis().set_ticklabels([])

    # Save the figure
    plt.tight_layout()
    # plt.savefig("kde_" + str(i).rjust(3, "0") + ".png", dpi=120)

    plt.show()
    print()

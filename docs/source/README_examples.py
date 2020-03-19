#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:20:48 2020

@author: tommy
"""

if __name__ == "__main__":
    from KDEpy import FFTKDE
    import matplotlib.pyplot as plt

    customer_ages = [40, 56, 20, 35, 27, 24, 29, 37, 39, 46]

    # Distribution of customers
    x, y = FFTKDE(bw="silverman").fit(customer_ages).evaluate()
    plt.plot(x, y)

    # Distribution of customer income (weight each customer by their income)
    customer_income = [152, 64, 24, 140, 88, 64, 103, 148, 150, 132]

    # The `bw` parameter can be manually set, e.g. `bw=5`
    x, y = FFTKDE(bw="silverman").fit(customer_ages, weights=customer_income).evaluate()
    plt.plot(x, y)


if __name__ == "__main__":
    plt.show()  # Flush out the calls to plt.plot earlier

    import os
    import numpy as np

    # Convert to arrays
    customer_ages = np.array(customer_ages)
    customer_income = np.array(customer_income)

    # Create the splot
    plt.figure(figsize=(10, 2.25))

    # ------------------------------------------------------------------------
    ax1 = plt.subplot(121)

    plt.title("Distribution of customers")
    x, y = FFTKDE(bw="silverman").fit(customer_ages).evaluate()
    plt.plot(x, y, zorder=10)
    plt.scatter(
        customer_ages, np.zeros_like(customer_ages), marker="o", color="red", zorder=10, s=np.min(customer_income / 2)
    )

    plt.grid(True, ls="--", zorder=-15)
    plt.yticks(fontsize=0)
    plt.xlabel("Age")
    plt.ylabel("Probability density")

    # ------------------------------------------------------------------------
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    plt.title("Distribution of customer income")
    x, y = FFTKDE(bw="silverman").fit(customer_ages, weights=customer_income).evaluate()
    plt.plot(x, y, zorder=10)
    plt.scatter(customer_ages, np.zeros_like(customer_ages), marker="o", color="red", zorder=10, s=customer_income / 2)

    plt.grid(True, ls="--", zorder=-15)
    plt.yticks(fontsize=0)
    plt.xlabel("Age")

    # ------------------------------------------------------------------------
    # Save the figure
    here = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(here, "_static/img", "README_example.png")
    plt.savefig(filename, dpi=200)

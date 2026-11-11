"""Finish a saved diagnostic figure in a notebook or batch run."""

import matplotlib.pyplot as plt


def show_or_close(fig, show_plots=True):
    if show_plots:
        plt.show()
    plt.close(fig)

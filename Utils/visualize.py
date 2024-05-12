import itertools
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1 - x[0]) ** 2 + 5 * (x[1] - x[0] ** 2) ** 2


def contour_plot(x_path, f_path, title):
    x = np.arange(-15, 15, 0.025)
    y = np.arange(-15, 15, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i, j in itertools.product(range(len(X)), range(len(X))):
        Z[i][j] = f(np.array([X[i][j], Y[i][j]]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_path[:, 0], x_path[:, 1], linestyle="--", marker="o", color="orange")
    ax.plot(x_path[-1, 0], x_path[-1, 1], "ro")
    ax.set(title=title, xlabel="x1", ylabel="x2")
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize="smaller", fmt="%1.2f")
    ax.axis("square")
    plt.tight_layout()
    plt.show()

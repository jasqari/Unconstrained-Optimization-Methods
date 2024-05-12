import math
import numpy as np
from Utils import visualize


def f(x):
    return (1 - x[0]) ** 2 + 5 * (x[1] - x[0] ** 2) ** 2


def gradient(x):
    dx0 = (20 * x[0] ** 3) - (20 * x[0] * x[1]) + (2 * x[0]) - 2
    dx1 = (10 * x[1]) - (10 * x[0] ** 2)
    return np.array([dx0, dx1], dtype=np.float64)


def hessian(x):
    dx00 = (60 * x[0] ** 2) - (20 * x[1]) + 2
    dx01 = -20 * x[0]
    dx10 = dx01
    dx11 = 10
    return np.array([[dx00, dx01], [dx10, dx11]], dtype=np.float64)


def dogleg(delta, g, B):
    p_B = -1 * np.linalg.solve(B, g)
    if np.linalg.norm(p_B) <= delta:
        return p_B

    p_U = -1 * (np.dot(g, g) / np.matmul(np.matmul(np.transpose(g), B), g)) * g
    if np.linalg.norm(p_U) >= delta:
        return delta * p_U / np.linalg.norm(p_U)

    term1 = (np.dot(p_U, p_B - p_U) ** 2 - np.dot(p_B - p_U, p_B - p_U)) * (
        np.dot(p_U, p_U) - delta**2
    )
    tau = (-1 * np.dot(p_U, p_B - p_U) + math.sqrt(term1)) / np.dot(p_B - p_U, p_B - p_U) + 1
    if tau < 1:
        return p_U * tau
    return p_U + (tau - 1) * (p_B - p_U)


x0 = np.array([-2, -2])
num_iterations = 10000
delta_max = 1
tolerance = 0.00001
eta = 0.15

x_i = x0
delta_i = 0.1
g_i = gradient(x_i)
iteration = 0
x_path = [x_i]
f_path = [f(x_i)]
while iteration < num_iterations and np.linalg.norm(g_i) > tolerance:
    g_i = gradient(x_i)
    B_i = hessian(x_i)
    p_i = dogleg(delta_i, g_i, B_i)

    actual_reduction = f(x_i) - f(x_i + p_i)
    predicted_reduction = (f(x_i) + 0 + 0) - (
        f(x_i) + np.dot(p_i, g_i) + 0.5 * np.dot(np.dot(p_i, B_i), p_i)
    )
    rho_i = actual_reduction / predicted_reduction
    if rho_i < 0.25:
        delta_i = 0.25 * delta_i
    else:
        if rho_i > 0.75 and np.linalg.norm(p_i) == delta_i:
            delta_i = min(2 * delta_i, delta_max)
        else:
            delta_i = delta_i

    if rho_i > eta:
        x_i = x_i + p_i
    else:
        x_i = x_i

    x_path.append(x_i)
    f_path.append(f(x_i))
    iteration += 1

print("x0:", x0)
print("f(x0):", f(x0))
print("\nNumber of iterations:", iteration)
print("\nx*:", x_i)
print("f(x*):", f(x_i))

visualize.contour_plot(np.array(x_path), np.array(f_path), "Trust-Region Dogleg")

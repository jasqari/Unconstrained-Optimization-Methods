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


def Armijo(x, p, alpha, c, rho):
    x_i = x
    p_i = p
    alpha_i = alpha
    armijo = f(x_i + alpha_i * p_i) <= f(x_i) + c * alpha_i * np.dot(gradient(x_i), p_i)
    while not armijo:
        alpha_i = rho * alpha_i
        armijo = f(x_i + alpha_i * p_i) <= f(x_i) + c * alpha_i * np.dot(gradient(x_i), p_i)
    return alpha_i


def Wolfe(x, p, alpha, c1, c2):
    x_i = x
    p_i = p
    alpha_i = alpha
    alpha_min = 0
    alpha_max = math.inf
    while True:
        armijo = f(x_i + alpha_i * p_i) <= f(x_i) + c1 * alpha_i * np.dot(gradient(x_i), p_i)
        curvature = np.dot(gradient(x_i + alpha_i * p_i), p_i) >= c2 * np.dot(gradient(x_i), p_i)
        if not armijo:
            alpha_max = alpha_i
            alpha_i = 0.1 * (alpha_min + alpha_max)
        elif not curvature:
            alpha_min = alpha_i
            if alpha_max == math.inf:
                alpha_i = 2 * alpha_i
            else:
                alpha_i = 0.1 * (alpha_min + alpha_max)
        else:
            break
    return alpha_i


def BFGS(s, y, h):  # Broyden-Fletcher-Goldfarb-Shanno Update
    gamma = 1 / np.dot(y, s)
    term1 = np.identity(s.shape[0]) - gamma * np.outer(s, y)
    term2 = np.identity(y.shape[0]) - gamma * np.outer(y, s)
    term3 = gamma * np.outer(s, s)
    h_next = np.matmul(np.matmul(term1, h), term2) + term3
    return h_next


def DFP(s, y, h):  # Davidon-Fletcher-Powell Update
    term1_1 = np.matmul(np.matmul(h, np.outer(y, y)), h)
    term1_2 = np.matmul(np.matmul(np.transpose(y), h), y)
    term1 = term1_1 / term1_2
    term2 = np.outer(s, s) / np.dot(y, s)
    h_next = h - term1 + term2
    return h_next


def BroydenFamily(phi, s, y, h):  # Broyden Class of Updates
    return (1 - phi) * BFGS(s, y, h) + (phi) * DFP(s, y, h)


x0 = np.array([-2, -2])
num_iterations = 10000
tolerance = 0.00001
alpha = 1

x_i = x0
alpha_i = alpha
h_i = np.linalg.inv(hessian(x_i))
iteration = 0
convergence = False
x_path = [x_i]
f_path = [f(x_i)]
while iteration < num_iterations and not convergence:
    x_i_prev = x_i
    h_i_prev = h_i
    try:
        p_i = -1 * np.matmul(h_i_prev, gradient(x_i_prev))
    except:
        print("Runtime Error: Singular matrix.\n")
        x_path = x_path[0:3]
        break
    alpha_i = Wolfe(x_i_prev, p_i, alpha_i, 10 ^ -4, 0.9)
    x_i = x_i_prev + alpha_i * p_i

    s_i = x_i - x_i_prev
    y_i = gradient(x_i) - gradient(x_i_prev)
    h_i = BroydenFamily(0, s_i, y_i, h_i_prev)  #

    x_path.append(x_i)
    f_path.append(f(x_i))
    convergence = math.isclose(np.linalg.norm(x_i_prev), np.linalg.norm(x_i), rel_tol=tolerance)
    iteration += 1

print("x0:", x0)
print("f(x0):", f(x0))
print("\nNumber of iterations:", iteration)
print("\nx*:", x_i)
print("f(x*):", f(x_i))


visualize.contour_plot(np.array(x_path), np.array(f_path), "Quasi-Newton")

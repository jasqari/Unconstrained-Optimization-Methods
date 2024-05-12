## Overview
In this repository, you can find the Python implementations of some line search and trust region approaches to unconstrained mathematical optimization.
* Gradient descent, conjugate gradient, Newton–Raphson, quasi-Newton, and Powell's dog approaches are implemented.
* Line search methods optimize using a dynamic step size based on the Wolfe conditions.
* `Utils/` contains a module that can display contour plots.
* The following multivariate function and its gradient are defined in all of the implementations:
<p align="center">
$f(\mathbf{x}) = (1-x_1)^2 + 5(x_2-x_1^2)^2$
</p>


## Usage
The parameters of the algorithms are hard-coded. Hence, you will have to manually set them according to your will. First, specify the starting point, initial step size, stopping criteria, etc., and then execute your chosen algorithm, for example:
```
python3 quasi_newton.py
```
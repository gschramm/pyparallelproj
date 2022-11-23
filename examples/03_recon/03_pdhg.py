import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import pyparallelproj.operators as operators
import pyparallelproj.functionals as functionals
import pyparallelproj.algorithms as algorithms

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

beta = 1e0
xp = np
n = 20
num_iter = 5000

noise = 'poisson'

#--------------------------------------------------------------------------

xp.random.seed(4)

x_true = xp.pad(xp.random.rand(n - 4), 2)

data_operator = operators.MatrixOperator(
    5 * xp.random.rand(2 * x_true.shape[0], x_true.shape[0]), xp)
prior_operator = operators.GradientOperator(data_operator.input_shape, xp)
prior_norm = functionals.L2L1Norm(xp)

x_fwd = data_operator.forward(x_true)

if noise == 'gaussian':
    data = x_fwd + 0.1 * x_fwd.mean() * xp.random.randn(
        *data_operator.output_shape)
    data_distance = functionals.SquaredL2NormDistance(data, xp)
elif noise == 'poisson':
    data = xp.random.poisson(x_fwd)
    data_distance = functionals.NegativePoissonLogLikelihood(data, xp)
else:
    raise ValueError

sigma = 0.9 / float(data_operator.norm())
tau = 0.9 / float(data_operator.norm())

pdhg = algorithms.PDHG(data_operator, data_distance, prior_operator,
                       prior_norm, beta, sigma, tau)

pdhg.run(num_iter, calculate_cost=True)

cost_func = lambda x: data_distance(data_operator.forward(
    x)) + beta * prior_norm(prior_operator.forward(x))

# use powell optimizer as reference
from scipy.optimize import fmin_powell

res = fmin_powell(cost_func, pdhg.x)

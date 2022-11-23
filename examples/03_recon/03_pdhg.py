"""1D demo for PDHG"""
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

beta = 3e-1
xp = np
n = 20
num_iter = 10000

noise = 'poisson'

#--------------------------------------------------------------------------

xp.random.seed(4)

x_true = xp.pad(5 * xp.repeat(xp.random.rand(n - 4), 3), 2)

data_operator = operators.MatrixOperator(
    xp.random.rand(2 * x_true.shape[0], x_true.shape[0]), xp)
prior_operator = operators.GradientOperator(data_operator.input_shape, xp)
prior_norm = functionals.L2L1Norm(xp)

x_fwd = data_operator.forward(x_true)

contamination = xp.full(x_fwd.shape, 0.5 * x_fwd.mean())

if noise == 'gaussian':
    data = x_fwd + contamination + 0.1 * x_fwd.mean() * xp.random.randn(
        *data_operator.output_shape)
    data_distance = functionals.SquaredL2NormDistance(data, xp)
elif noise == 'poisson':
    data = xp.random.poisson(x_fwd + contamination)
    data_distance = functionals.NegativePoissonLogLikelihood(data, xp)
else:
    raise ValueError

sigma = 0.9 / float(data_operator.norm())
tau = 0.9 / float(data_operator.norm())

pdhg = algorithms.PDHG(data_operator,
                       data_distance,
                       prior_operator,
                       prior_norm,
                       beta,
                       sigma,
                       tau,
                       contamination=contamination)

pdhg.run(num_iter, calculate_cost=True)

cost_func = lambda x: data_distance(data_operator.forward(
    x) + contamination) + beta * prior_norm(prior_operator.forward(x))

# use powell optimizer as reference
from scipy.optimize import fmin_powell

res = fmin_powell(cost_func, pdhg.x)

fig, ax = plt.subplots()
ax.plot(pdhg.x, marker='o')
ax.plot(res, '.')
fig.tight_layout()
fig.show()
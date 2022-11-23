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

beta = 0.3
xp = np
n = 20
lower_bound = 0.
upper_bound = xp.inf
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
                       contamination=contamination,
                       g_functional=functionals.BoundIndicatorFunctional(
                           xp, lb=lower_bound, ub=upper_bound))

pdhg.run(num_iter, calculate_cost=True)

cost_func = lambda x: data_distance(data_operator.forward(
    x) + contamination) + beta * prior_norm(prior_operator.forward(x))

# use SLSQP optimizer stared from PDHG solution to see if we can get a better solution
from scipy.optimize import minimize, Bounds

bounds = Bounds(lb=np.full(x_true.shape, lower_bound),
                ub=np.full(x_true.shape, upper_bound))

res = minimize(cost_func, pdhg.x, method='SLSQP', bounds=bounds)

print(f'fmin PDHG  ..: {pdhg.cost[-1]}')
print(f'fmin SLSQP ..: {res.fun}')

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
it = np.arange(1, num_iter + 1)
ax[0].plot(pdhg.x, marker='o')
ax[0].plot(res.x, '.')
ax[1].plot(it, pdhg.cost)
ax[2].plot(it[5:200], pdhg.cost[5:200])
ax[3].plot(it[-1000:], pdhg.cost[-1000:])
fig.tight_layout()
fig.show()
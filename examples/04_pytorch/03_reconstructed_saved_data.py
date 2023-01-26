"""script to check whether saved 2D OSEM data can be reconstructed correctly"""
import dill
import json
import cupy as cp
import matplotlib.pyplot as plt

from pathlib import Path

import pyparallelproj.algorithms as algorithms

seed = 1
odir = Path('../data/OSEM_2D_5.00E+02/004_000_032/')

#----------------------------------------------------------
# load the data

with open(odir / 'parameters.json', 'r') as f:
    parameters = json.load(f)

image = cp.load(odir / 'image.npy')
osem = cp.load(odir / f'osem_{seed:03}.npy')
data = cp.load(odir / f'data_{seed:03}.npy')
image_fwd = cp.load(odir / 'image_fwd.npy')
multiplicative_corrections = cp.load(odir / 'multiplicative_corrections.npy')
contamination = cp.load(odir / 'contamination.npy')

with open(odir / 'projector.pkl', 'rb') as f:
    projector = dill.load(f)

#----------------------------------------------------------
# rerun the recon

# the projector gets saved without multiplicative corrections
# (more efficient for torch batch workflow)
# so we have to set them again

projector.multiplicative_corrections = multiplicative_corrections
reconstructor = algorithms.OSEM(data, contamination, projector, verbose=True)
reconstructor.run(parameters['num_iterations'], evaluate_cost=False)

x = reconstructor.x

# check if saved OSEM is the same as our recon
assert (cp.all(cp.isclose(x, osem)))

#----------------------------------------------------------
# show the results

ikws = dict(vmin=0, vmax=1.2 * float(image.max()))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(cp.asnumpy(image), **ikws)
ax[1].imshow(cp.asnumpy(osem), **ikws)
ax[2].imshow(cp.asnumpy(x), **ikws)
fig.tight_layout()
fig.show()
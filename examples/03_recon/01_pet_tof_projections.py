import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

xp = cp

num_rings = 1
symmetry_axis = 2

num_subsets = 1

voxsize = (2., 2., 2.)

sinogram_order = 'RVP'

num_trans = 200
num_ax = 1
img_shape = (num_trans, num_trans, num_ax)

#---------------------------------------------------------------------

img_origin = ((-0.5 * num_trans + 0.5) * voxsize[0],
              (-0.5 * num_trans + 0.5) * voxsize[1],
              (-0.5 * num_ax + 0.5) * voxsize[2])

img = xp.zeros(img_shape, dtype=xp.float32)
img[(num_trans // 4):(-num_trans // 4),
    (num_trans // 4):(-num_trans // 4), :] = 1

# setup the scanner geometry
scanner = scanners.GEDiscoveryMI(num_rings, symmetry_axis=symmetry_axis, xp=xp)

# setup the coincidence descriptor
coincidence_descriptor = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=65,
    max_ring_difference=scanner.num_rings - 1,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order])

subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, num_subsets)

projector = petprojectors.PETJosephProjector(coincidence_descriptor, img_shape,
                                             img_origin, voxsize)
projector.subsetter = subsetter
projector.tof_parameters = tof.ge_discoverymi_tof_parameters

# simulate data
# Ax
img_fwd = projector.forward(img)
# additive contamination s
contamination = xp.ones_like(img_fwd)
contamination *= (0.5 * img_fwd.sum() / contamination.sum())

# data is Poisson(Ax + s)
sensitivity = 0.3
data = xp.random.poisson(sensitivity * (img_fwd + contamination))

# calculate the gradient of a random image with respect to the data fidelity term

random_img = xp.random.rand(*img_shape).astype(xp.float32)
random_img_fwd = sensitivity * (projector.forward(random_img) + contamination)

data_fidelity_gradient = projector.adjoint(
    sensitivity * (1 - data / random_img_fwd).astype(xp.float32))

#----------------------------------------------------
#----------------------------------------------------
# visualize results

# get arrays from GPU
if xp.__name__ == 'cupy':
    img = xp.asnumpy(img)
    img_fwd = xp.asnumpy(img_fwd)
    data = xp.asnumpy(data)
    data_fidelity_gradient = xp.asnumpy(data_fidelity_gradient)

# reshape img_fwd into a sinogram with 3 axis
tofsino_shape = (coincidence_descriptor.sinogram_spatial_shape) + (
    projector.tof_parameters.num_tofbins, )
img_fwd = img_fwd.reshape(tofsino_shape)
data = data.reshape(tofsino_shape)

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
ax[0].imshow(img[..., 0])
ax[1].imshow(img_fwd[..., 0, 10])
ax[2].imshow(data[..., 0, 10])
ax[3].imshow(data_fidelity_gradient[..., 0])
ax[0].set_title('x')
ax[1].set_title('A x (central TOF bin)')
ax[2].set_title('Poisson(A x)')
ax[3].set_title('data_fidelity_gradient')
fig.tight_layout()
fig.show()
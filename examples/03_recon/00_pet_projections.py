import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.petprojectors as petprojectors

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

xp = cp

radius = 350
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 1

max_ring_difference = num_rings - 1
radial_trim = 49

ring_positions = 5.55 * (np.arange(num_rings) - num_rings / 2 + 0.5)

num_subsets = 1

voxsize = (2., 2., 2.)

sinogram_order = 'RVP'
symmetry_axis = 2

num_trans = 200
img_shape = (num_trans, num_trans, 1)
img_origin = ((-0.5 * num_trans + 0.5) * voxsize[0],
              (-0.5 * num_trans + 0.5) * voxsize[1], 0.)
img = xp.zeros(img_shape, dtype=xp.float32)
img[(num_trans // 4):(-num_trans // 4),
    (num_trans // 4):(-num_trans // 4), :] = 1
#---------------------------------------------------------------------
scanner = scanners.RegularPolygonPETScannerGeometry(
    radius,
    num_sides,
    num_lor_endpoints_per_side,
    lor_spacing,
    num_rings,
    ring_positions,
    symmetry_axis=symmetry_axis,
    xp=xp)

# setup the coincidence descriptor
coincidence_descriptor = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order])

subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, num_subsets)

projector = petprojectors.NonTOFPETJosephProjector(coincidence_descriptor,
                                                   img_shape, img_origin,
                                                   voxsize, subsetter)

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
img_fwd = img_fwd.reshape(coincidence_descriptor.sinogram_spatial_shape)
data = data.reshape(coincidence_descriptor.sinogram_spatial_shape)

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
ax[0].imshow(img[..., 0])
ax[1].imshow(img_fwd[..., 0])
ax[2].imshow(data[..., 0])
ax[3].imshow(data_fidelity_gradient[..., 0])
ax[0].set_title('x')
ax[1].set_title('A x')
ax[2].set_title('Poisson(A x)')
ax[3].set_title('data_fidelity_gradient')
fig.tight_layout()
fig.show()
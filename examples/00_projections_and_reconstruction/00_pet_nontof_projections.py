import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.coincidences as coincidences
import pyparallelproj.petprojectors as petprojectors

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

# numpy / cupy module to use
xp = cp

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- input parmeters -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# scanner parameters
num_rings = 1
symmetry_axis = 2

# image parameters
voxsize = (2., 2., 2.)
num_trans = 200
num_ax = 1

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- setup a square test image ---------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

img_shape = (num_trans, num_trans, num_ax)
img_origin = ((-0.5 * num_trans + 0.5) * voxsize[0],
              (-0.5 * num_trans + 0.5) * voxsize[1],
              (-0.5 * num_ax + 0.5) * voxsize[2])

img = xp.zeros(img_shape, dtype=xp.float32)
img[(num_trans // 4):(-num_trans // 4),
    (num_trans // 4):(-num_trans // 4), :] = 1

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- setup the PET scanner (coicidence descriptor) -------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
    num_rings=num_rings,
    sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder['RVP'],
    xp=xp)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- setup the PET projector (linear forward operator) ---------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

projector = petprojectors.PETJosephProjector(coincidence_descriptor, img_shape,
                                             img_origin, voxsize)

# generate a random multiplicative correction sinogram simulating constant sensitivity
projector.multiplicative_corrections = xp.full(projector.output_shape, 0.3)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- simulate acquired data ------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# forward model A x
img_fwd = projector.forward(img)
# additive contamination s
contamination = xp.ones_like(img_fwd)
contamination *= (0.5 * img_fwd.sum() / contamination.sum())

# data is Poisson(Ax + s)
data = xp.random.poisson(img_fwd + contamination)

# calculate the gradient of a random image with respect to the data fidelity term
random_img = xp.random.rand(*img_shape).astype(xp.float32)
random_img_fwd = projector.forward(random_img) + contamination

data_fidelity_gradient = projector.adjoint(
    (1 - data / random_img_fwd).astype(xp.float32))

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- visualizations --------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

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
ax[2].set_title('Poisson(A x + s)')
ax[3].set_title('data_fidelity_gradient')
fig.tight_layout()
fig.show()
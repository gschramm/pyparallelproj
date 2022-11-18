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

xp = np

radius = 350
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 1

max_ring_difference = num_rings - 1
radial_trim = 49

ring_positions = 5.55 * (np.arange(num_rings) - num_rings / 2 + 0.5)

num_subsets = 5

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

attenuation_img = (0.01 * (img > 0)).astype(xp.float32)

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

# setup a non-time-of-flight and time-of-flight projector
nontof_projector = petprojectors.NonTOFPETJosephProjector(
    coincidence_descriptor, img_shape, img_origin, voxsize, subsetter)

tof_parameters = tof.TOFParameters(num_tofbins=27,
                                   tofbin_width=22.2,
                                   sigma_tof=60 / 2.35)

projector = petprojectors.TOFPETJosephProjector(coincidence_descriptor,
                                                img_shape, img_origin, voxsize,
                                                subsetter, tof_parameters)

# simulate the attenuation factors (exp(-fwd(attenuation_image)))
attenuation_factors = xp.exp(-nontof_projector.forward(attenuation_img))

# if the projector is a TOF projector, we have to expand the dimension of the
# nontof attenuation factors to be able to mulitply nontof and tof data
if len(projector.output_shape) > 1:
    attenuation_factors = xp.expand_dims(attenuation_factors, -1)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape, 1e-3)

# evaluate the full forward model
img_fwd = attenuation_factors * projector.forward(img) + contamination

# evaluated the forward model for a subset of all LORs
subset = 0
lors = projector.subsetter.get_subset_indices(subset)
img_fwd_subset = attenuation_factors[lors] * projector.forward_subset(
    img, lors=lors) + contamination[lors]

#q = img_fwd.reshape(
#    subsetter.get_sinogram_subset_shape(subset) +
#    (tof_parameters.num_tofbins, ))

## simulate data
## Ax
#img_fwd = projector.forward(img)
## additive contamination s
#contamination = xp.ones_like(img_fwd)
#contamination *= (0.5 * img_fwd.sum() / contamination.sum())
#
## data is Poisson(Ax + s)
#sensitivity = 0.3
#data = xp.random.poisson(sensitivity * (img_fwd + contamination))
#
## calculate the gradient of a random image with respect to the data fidelity term
#
#random_img = xp.random.rand(*img_shape).astype(xp.float32)
#random_img_fwd = sensitivity * (projector.forward(random_img) + contamination)
#
#data_fidelity_gradient = projector.adjoint(
#    sensitivity * (1 - data / random_img_fwd).astype(xp.float32))
#
##----------------------------------------------------
##----------------------------------------------------
## visualize results
#
## get arrays from GPU
#if xp.__name__ == 'cupy':
#    img = xp.asnumpy(img)
#    img_fwd = xp.asnumpy(img_fwd)
#    data = xp.asnumpy(data)
#    data_fidelity_gradient = xp.asnumpy(data_fidelity_gradient)
#
## reshape img_fwd into a sinogram with 3 axis
#tofsino_shape = (coincidence_descriptor.sinogram_spatial_shape) + (
#    tof_parameters.num_tofbins, )
#img_fwd = img_fwd.reshape(tofsino_shape)
#data = data.reshape(tofsino_shape)
#
#fig, ax = plt.subplots(1, 4, figsize=(12, 3))
#ax[0].imshow(img[..., 0])
#ax[1].imshow(img_fwd[..., 0, 10])
#ax[2].imshow(data[..., 0, 10])
#ax[3].imshow(data_fidelity_gradient[..., 0])
#ax[0].set_title('x')
#ax[1].set_title('A x')
#ax[2].set_title('Poisson(A x)')
#ax[3].set_title('data_fidelity_gradient')
#fig.tight_layout()
#fig.show()
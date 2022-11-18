import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.acquisition_models as acquisition_models

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

xp = cp

#-------------------
# scanner parameters
radius = 350
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 18

max_ring_difference = num_rings - 1
radial_trim = 49

ring_positions = 5.55 * (np.arange(num_rings) - num_rings / 2 + 0.5)
symmetry_axis = 2

#-------------------
# image parameters
num_trans = 200
voxsize = (2., 2., 2.)

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# reconstruction parameters
num_iterations = 4
num_subsets = 28

# global sensitivity factor of the scanner that can be used to
# control the number of simulated counts
scanner_sensitivty = 0.1
#---------------------------------------------------------------------

num_axial = max(
    int((ring_positions.max() - ring_positions.min()) /
        voxsize[symmetry_axis]), 1)

img_shape = (num_trans, num_trans, num_axial)

img_origin = ((-0.5 * num_trans + 0.5) * voxsize[0],
              (-0.5 * num_trans + 0.5) * voxsize[1], 0.)
img = xp.zeros(img_shape, dtype=xp.float32)

# assign random value to central square
img[(num_trans // 4):(-num_trans // 4),
    (num_trans // 4):(-num_trans // 4), :] = 4.3

attenuation_img = (0.01 * (img > 0)).astype(xp.float32)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
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

# simulate LOR sensitivity factors
sensitivity_factors = xp.full(nontof_projector.output_shape,
                              scanner_sensitivty,
                              dtype=xp.float32)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape, 1e-3, dtype=xp.float32)

# setup the forward operator ("A") that also supports subsets
acq_model = acquisition_models.PETAcquisitionModel(projector,
                                                   attenuation_factors,
                                                   sensitivity_factors)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# simulate acquired data based on forward model and known contaminations

img_fwd = acq_model.forward(img)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape,
                        img_fwd.mean() / 10.,
                        dtype=xp.float32)

# generate noisy data
data = xp.random.poisson(img_fwd + contamination).astype(xp.uint16)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# run an OSEM reconstruction

# (1) calculate the sensitivity images for all subsets
sensitivity_images = xp.zeros(
    (acq_model.subsetter.num_subsets, ) + acq_model.input_shape,
    dtype=xp.float32)

for subset in range(acq_model.subsetter.num_subsets):
    ones = xp.ones(acq_model.get_subset_shape(subset), dtype=xp.float32)
    sensitivity_images[subset, ...] = acq_model.adjoint_subset(ones, subset)

# intialize the reconstruction
x_init = xp.ones(acq_model.input_shape, dtype=xp.float32)
x = x_init.copy()

# run OSEM updates
for it in range(num_iterations):
    print(f'OSEM iteration {(it+1):03}')
    for subset in range(acq_model.subsetter.num_subsets):
        # get the LOR indices belonging to the current subset
        inds = acq_model.subsetter.get_subset_indices(subset)
        # calculate the expected data given the current reconstruction
        expected_data = acq_model.forward_subset(
            x, inds=inds) + contamination[inds]
        # ratio of measured data and expected data
        ratio = (data[inds] / expected_data).astype(xp.float32)
        # OSEM update
        x *= (acq_model.adjoint_subset(ratio, inds=inds) /
              sensitivity_images[subset, ...])

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# visualizations

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    data = xp.asnumpy(data)
    x = xp.asnumpy(x)
    img = xp.asnumpy(img)

# reshape the data into a sinogram (just for visualizations)
data_reshaped = data.reshape(coincidence_descriptor.sinogram_spatial_shape +
                             (tof_parameters.num_tofbins, ))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ims = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * img.max())
ax[0].imshow(img[:, :, 0], **ims)
ax[1].imshow(x[:, :, 0], **ims)
ax[2].imshow(data_reshaped[:, :, num_rings // 2,
                           tof_parameters.num_tofbins // 2],
             cmap=plt.cm.Greys)
fig.tight_layout()
fig.show()
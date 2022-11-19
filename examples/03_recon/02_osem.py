import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.acquisition_models as acquisition_models
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

xp = cp

if xp.__name__ == 'cupy':
    import cupyx.scipy.ndimage as ndi
else:
    import scipy.ndimage as ndi

#-------------------
# scanner parameters
radius = 350
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 1

max_ring_difference = num_rings - 1
radial_trim = 49

ring_positions = 5.55 * (np.arange(num_rings) - num_rings / 2 + 0.5)
symmetry_axis = 2

fwhm_mm = 4.5

#-------------------
# image parameters
num_trans = 200
voxsize = (2., 2., 2.)

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# reconstruction parameters
num_iterations = 10
num_subsets = 28

# global sensitivity factor of the scanner that can be used to
# control the number of simulated counts
scanner_sensitivty = 1.

#---------------------------------------------------------------------

num_axial = max(
    int((ring_positions.max() - ring_positions.min()) /
        voxsize[symmetry_axis]), 1)

img_shape = (num_trans, num_trans, num_axial)

img_origin = ((-0.5 * num_trans + 0.5) * voxsize[0],
              (-0.5 * num_trans + 0.5) * voxsize[1],
              (-0.5 * num_axial + 0.5) * voxsize[2])

img = xp.zeros(img_shape, dtype=xp.float32)

# assign random value to central square
img[(num_trans // 5):(-num_trans // 5),
    (num_trans // 5):(-num_trans // 5), :] = 4.3
img[(num_trans // 3):(-num_trans // 3),
    (num_trans // 3):(-num_trans // 3), :] = 6

attenuation_img = (0.01 * (img > 0)).astype(xp.float32)

res_model = resolution_models.GaussianImageBasedResolutionModel(
    img_shape, tuple(fwhm_mm / (2.35 * x) for x in voxsize), xp, ndi)
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
acq_model = acquisition_models.PETAcquisitionModel(
    projector,
    attenuation_factors,
    sensitivity_factors,
    image_based_resolution_model=res_model)

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

reconstructor = algorithms.OSEM(data, contamination, acq_model, verbose=True)
reconstructor.run(num_iterations, evaluate_cost=False)

x = reconstructor.x

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
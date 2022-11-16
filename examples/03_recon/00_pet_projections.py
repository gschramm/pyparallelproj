import numpy as np

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
num_rings = 5

max_ring_difference = num_rings - 1
radial_trim = 49

ring_positions = 5.55 * (np.arange(num_rings) - num_rings / 2 + 0.5)

num_subsets = 1

voxsize = (2., 2., 2.)
num_trans = 300

sinogram_order = 'RVP'
symmetry_axis = 2

img_shape = (123, 45, 5)
img_origin = (0., 0., 0.)
img = xp.ones(img_shape, dtype=xp.float32)
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

img_fwd = projector.forward(img)
img_fwd_back = projector.adjoint(img_fwd)

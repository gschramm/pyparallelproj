import time

import numpy as np

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.wrapper as ppw
import pyparallelproj.subsets as subsets

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

#---------------------------------------------------------------------

xp = np
sinogram_order = coincidences.SinogramSpatialAxisOrder.PVR
symmetry_axis = 1

radius = 330
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 45

max_ring_difference = num_rings - 1
radial_trim = 1

ring_positions = lor_spacing * (np.arange(num_rings) - num_rings / 2 + 0.5)

num_subsets = 28
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
cd = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=sinogram_order)

subsetter = subsets.SingoramViewSubsetter(cd, num_subsets)

#----------------------------------------------------------------------------
lors = subsetter.get_subset_indices(0)

start_mod, start_ind, end_mod, end_ind = cd.get_lor_indices(lors)
xstart = scanner.get_lor_endpoints(start_mod, start_ind).astype(xp.float32)
xend = scanner.get_lor_endpoints(end_mod, end_ind).astype(xp.float32)

# setup a box like test image
voxsize = np.array([2., 2., 2.], dtype=np.float32)
img_shape = [250, 250, 250]
img_shape[symmetry_axis] = int(
    (ring_positions.max() - ring_positions.min()) / voxsize[0])
img_shape = tuple(img_shape)
n0, n1, n2 = img_shape

# setup an image containing a square
img = xp.zeros(img_shape, dtype=np.float32)
img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4),
    (n2 // 4):(3 * n2 // 4)] = 1

# setup the image origin = the coordinate of the [0,0,0] voxel
img_origin = (-(np.array(img.shape, dtype=np.float32) / 2) + 0.5) * voxsize
img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)

print(cd.sinogram_spatial_axis_order.name)
print(symmetry_axis, img_shape)

# perform a complete fwd projection
t0 = time.time()
ppw.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)
if xp.__name__ == 'cupy':
    cp.cuda.Device().synchronize()
t1 = time.time()
t_fwd = t1 - t0
print(t_fwd)

# perform a complete backprojection
back_img = xp.zeros(img.shape, dtype=xp.float32)
t2 = time.time()
ppw.joseph3d_back(xstart, xend, back_img, img_origin, voxsize, img_fwd)
if xp.__name__ == 'cupy':
    cp.cuda.Device().synchronize()
t3 = time.time()
t_back = t3 - t2
print(t_back)
print(t_fwd + t_back)

img_fwd = img_fwd.reshape(subsetter.get_sinogram_subset_shape(0))
if xp.__name__ == 'cupy':
    img_fwd = cp.asnumpy(img_fwd)
    back_img = cp.asnumpy(back_img)

##print(timeit.timeit('ppw.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)', globals=globals()), number = 1)
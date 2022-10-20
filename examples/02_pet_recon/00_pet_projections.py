import timeit
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanner_modules as pps
import pyparallelproj.data as ppd
import pyparallelproj.wrapper as ppw

try:
    import cupy as cp
except:
    warning.warn('cupy module not available')
    import numpy as cp

#---------------------------------------------------------------------

xp = cp
sinogram_order = ppd.SinogramSpatialAxisOrder.PVR

radius = 330
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 45

max_ring_difference = num_rings - 1

radial_trim = 1

ring_positions = lor_spacing * (np.arange(num_rings) - num_rings/2 + 0.5)

#---------------------------------------------------------------------

scanner = pps.RegularPolygonPETScannerGeometry(radius,
                                               num_sides,
                                               num_lor_endpoints_per_side,
                                               lor_spacing,
                                               num_rings,
                                               ring_positions,
                                               symmetry_axis=0,
                                               xp = xp)


# setup the coincidence descriptor
cd = ppd.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=sinogram_order)

#----------------------------------------------------------------------------
lors = np.arange(cd.num_lors)
start_mod, start_ind, end_mod, end_ind = cd.get_lor_indices(lors)
xstart = scanner.get_lor_endpoints(start_mod, start_ind).astype(np.float32)
xend = scanner.get_lor_endpoints(end_mod, end_ind).astype(np.float32)

# setup a box like test image
voxsize = np.array([2., 2., 2.], dtype=np.float32)
n0 = int((ring_positions.max() - ring_positions.min()) / voxsize[0])
n1 = 200
n2 = 200

# setup an image containing a square
img = xp.zeros((n0, n1, n2), dtype=np.float32)
img[(n0 // 4):(3 * n0 // 4),
    (n1 // 4):(3 * n1 // 4),
    (n2 // 4):(3 * n2 // 4)] = 1

# setup the image origin = the coordinate of the [0,0,0] voxel
img_origin = (-(np.array(img.shape, dtype=np.float32) / 2) + 0.5) * voxsize
img_fwd = xp.zeros(xstart.shape[0], dtype=np.float32)

print(cd.sinogram_spatial_axis_order.name)

# perform a complete fwd projection
t0 = time.time()
ppw.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)
if xp.__name__ == 'cupy':
    cp.cuda.Device().synchronize()
t1 = time.time()
print(t1-t0)

# perform a complete backprojection
back_img = xp.zeros(img.shape, dtype=xp.float32)
t2 = time.time()
ppw.joseph3d_back(xstart, xend, back_img, img_origin, voxsize, img_fwd)
if xp.__name__ == 'cupy':
    cp.cuda.Device().synchronize()
t3 = time.time()
print(t3-t2)

if xp.__name__ == 'cupy':
    img_fwd = cp.asnumpy(img_fwd).reshape(cd.sinogram_spatial_shape)
    back_img = cp.asnumpy(back_img)


##print(timeit.timeit('ppw.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)', globals=globals()), number = 1)

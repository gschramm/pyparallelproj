import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences

radius = 330
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 27

max_ring_difference = num_rings - 1

radial_trim = 49

ring_positions = lor_spacing * (np.arange(num_rings) - num_rings / 2 + 0.5)

scanner = scanners.RegularPolygonPETScannerGeometry(radius,
                                                    num_sides,
                                                    num_lor_endpoints_per_side,
                                                    lor_spacing,
                                                    num_rings,
                                                    ring_positions,
                                                    symmetry_axis=0)

# setup the coincidence descriptor
cd = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder.PVR)

#----------------------------------------------------------------------------
# get the start / end module and index number for the first 3 lors
all_lors = np.arange(cd.num_lors)
# choose 200 random lors
lors = np.random.choice(all_lors, 200)

start_mod, start_ind, end_mod, end_ind = cd.get_lor_indices(lors)
xstart = scanner.get_lor_endpoints(start_mod, start_ind).astype(np.float32)
xend = scanner.get_lor_endpoints(end_mod, end_ind).astype(np.float32)

#--------------------------------------------------------------------------

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax,
                           show_linear_index=False,
                           annotation_fontsize=0,
                           s=1)

# show all LORs of a views in a single plane
cd.show_view(ax, cd.num_views // 2, scanner.num_rings // 2)

fig.tight_layout()
fig.show()

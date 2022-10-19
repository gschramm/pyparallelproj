import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanner_modules as pps
import pyparallelproj.data as ppd

radius = 50
num_sides = 12
num_lor_endpoints_per_side = 6
lor_spacing = 4.
num_rings = 4

max_ring_difference = 2
radial_trim = 10

ring_positions = 1.1 * lor_spacing * np.arange(num_rings)

scanner = pps.RegularPolygonPETScannerGeometry(radius,
                                               num_sides,
                                               num_lor_endpoints_per_side,
                                               lor_spacing,
                                               num_rings,
                                               ring_positions,
                                               symmetry_axis=0)

# setupt the coincidence descriptor
cd = ppd.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=ppd.SinogramSpatialAxisOrder.RVP)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax,
                           show_linear_index=False,
                           annotation_fontsize=0,
                           s=1)

# get the start / end module and index number for the first 3 lors
nfirst = 3
lors = np.arange(nfirst)
start_mod, start_ind, end_mod, end_ind = cd.get_lor_indices(lors)
start_coords = scanner.get_lor_endpoints(start_mod, start_ind)
end_coords = scanner.get_lor_endpoints(end_mod, end_ind)

print(f'\nstart coordinates of first {nfirst} LORs')
print(start_coords)
print(f'\nend coordinates of first {nfirst} LORs')
print(end_coords)

# show two "views" in a single plane
cd.show_view(ax, 0, 0)
cd.show_view(ax, cd.num_views // 2, num_rings - 1, color='r')

## show the first 3 LORs
#cd.show_lors(np.arange(3), ax)
## show the last 3 LORs
#cd.show_lors(np.arange(cd.num_lors - 3, cd.num_lors), ax, color='r')
fig.tight_layout()
fig.show()

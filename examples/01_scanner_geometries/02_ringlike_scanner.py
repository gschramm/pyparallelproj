from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scanner_modules as pps

radius = 50
num_sides = 7
num_lor_endpoints_per_side = 4
lor_spacing = 4.
num_rings = 3

max_ring_difference = 1
radial_trim = 5

ring_positions = 1.1 * lor_spacing * np.arange(num_rings)

scanner = pps.RegularPolygonPETScannerGeometry(radius,
                                               num_sides,
                                               num_lor_endpoints_per_side,
                                               lor_spacing,
                                               num_rings,
                                               ring_positions,
                                               symmetry_axis=0)

# setupt the coincidence descriptor
cd = pps.RegularPolygonPETCoincidenceDescriptor(
    scanner, radial_trim=radial_trim, max_ring_difference=max_ring_difference)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax,
                           show_linear_index=False,
                           annotation_fontsize=0,
                           s=1)
cd.show_view(ax, 0, num_rings // 2)
fig.tight_layout()
fig.show()
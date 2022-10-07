# num_rings
# lor_endpoint_ring_number
# lor_endpoint_index_in_ring
# lor_endpoint_module_in_ring
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scanner_modules as pps

radius = 350.
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 9 * 3

ring_positions = 1.2 * lor_spacing * np.arange(num_rings)

scanner = pps.RegularPolygonPETScanner(radius,
                                       num_sides,
                                       num_lor_endpoints_per_side,
                                       lor_spacing,
                                       num_rings,
                                       ring_positions,
                                       symmetry_axis=0)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax,
                           show_linear_index=False,
                           annotation_fontsize=0,
                           s=1)
#scanner.show_all_lors_for_endpoint(ax, 27, 2)
fig.tight_layout()
fig.show()
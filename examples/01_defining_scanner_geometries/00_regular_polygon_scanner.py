"""example script to show how to setup a PET scanner consisting of stacked regular polygons (rings)
   this should cover most of the existing commercial WB PET scanners"""
import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences

#--------------------------------------------------------------------------------
# scanner parameters

radius = 330
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
num_rings = 27
max_ring_difference = num_rings - 1
radial_trim = 49
ring_positions = lor_spacing * (np.arange(num_rings) - num_rings / 2 + 0.5)

#--------------------------------------------------------------------------------

scanner = scanners.RegularPolygonPETScannerGeometry(radius,
                                                    num_sides,
                                                    num_lor_endpoints_per_side,
                                                    lor_spacing,
                                                    num_rings,
                                                    ring_positions,
                                                    symmetry_axis=0)

# setup a coincidence descriptor that describes which LOR endpoints are connected
# in case of a ring-like scanner the LORs can be ordered into a sinogram with
# the a "plane", "view" and "radial" direction
cd = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder.PVR)

#--------------------------------------------------------------------------
# show all LORs endpoints of the scanner and a few LORs
# defining one sinogram view in one sinogram plane

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
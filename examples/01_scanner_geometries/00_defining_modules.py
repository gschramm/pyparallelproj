"""script that shows how to define a scanner consiting of multiple rings of
   regular polygons"""
import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scannermodules as pps

radius = 20.
num_sides = 7
lor_spacing = 4.
ring_spacing = 5.
num_lor_endpoints_per_side = 3
num_rings = 4

phis = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)

mods = []

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')

#-------------------------------------------------------
# this is the way that RegularPolygonPETScannerModule defines the modules
# here a module is a "ring"

for i in range(num_rings):
    aff_mat2 = np.eye(4)
    aff_mat2[0, -1] = (i - num_rings / 2 + 0.5) * ring_spacing

    mods.append(
        pps.RegularPolygonPETScannerModule(
            radius,
            num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            affine_transformation_matrix=aff_mat2))

    mods[-1].show_lor_endpoints(ax,
                                annotation_fontsize=6.,
                                annotation_prefix=f'{i},')

fig.tight_layout()
fig.show()
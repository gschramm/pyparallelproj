import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scanner_modules as pps

radius = 20.
num_sides = 7
lor_spacing = (5., 4.)
num_lor_endpoints_per_side = 3
num_rings = 4

phis = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)

mods = []

fig = plt.figure(figsize=(14, 7))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

for i, phi in enumerate(phis):
    aff_mat = np.eye(4)
    aff_mat[:-1, :-1] = Rotation.from_euler(
        'xyz',
        [-phi, 0, 0],
    ).as_matrix()

    aff_mat[1, -1] = np.sin(phi) * radius
    aff_mat[2, -1] = np.cos(phi) * radius

    mods.append(
        pps.RectangularPETScannerModule(
            (num_lor_endpoints_per_side, num_rings),
            lor_spacing,
            ax0=1,
            ax1=0,
            affine_transformation_matrix=aff_mat))

    mods[-1].show_lor_endpoints(ax0,
                                annotation_fontsize=6.,
                                annotation_prefix=f'{i},')

#------------

mods2 = []

for i in range(num_rings):
    aff_mat2 = np.eye(4)
    aff_mat2[0, -1] = (i - num_rings / 2 + 0.5) * lor_spacing[1]

    mods2.append(
        pps.RegularPolygonPETScannerModule(
            radius,
            num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            affine_transformation_matrix=aff_mat2))

    mods2[-1].show_lor_endpoints(ax1,
                                 annotation_fontsize=6.,
                                 annotation_prefix=f'{i},')

fig.tight_layout()
fig.show()
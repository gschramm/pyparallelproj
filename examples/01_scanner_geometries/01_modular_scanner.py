"""a short script showing how to define an ringlike scanner with an extra module"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scannermodules as scannermods
import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences

radius = 20.
num_sides = 7
lor_spacing = 4.
ring_spacing = 5.
num_lor_endpoints_per_side = 3
num_rings = 4

phis = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)

mods = []

for i in range(num_rings):
    aff_mat2 = np.eye(4)
    aff_mat2[0, -1] = (i - num_rings / 2 + 0.5) * ring_spacing

    mods.append(
        scannermods.RegularPolygonPETScannerModule(
            radius,
            num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            affine_transformation_matrix=aff_mat2))

# add extra module outside the ring
aff_mat = np.eye(4)
aff_mat[:-1, :-1] = Rotation.from_euler(
    'xyz',
    [0, np.pi / 2, 0],
).as_matrix()
aff_mat[0, -1] = -0.75 * num_rings * ring_spacing

mods.append(
    scannermods.RectangularPETScannerModule(
        (5, 5), (lor_spacing, lor_spacing),
        ax0=1,
        ax1=0,
        affine_transformation_matrix=aff_mat))

#------------
mods = tuple(mods)

scanner = scanners.ModularizedPETScannerGeometry(mods)
cd = coincidences.GenericPETCoincidenceDescriptor(scanner)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax, show_linear_index=False, annotation_fontsize=6)
cd.show_all_lors_for_endpoint(ax, 1, 1)
fig.tight_layout()
fig.show()
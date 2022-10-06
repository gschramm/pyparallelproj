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

# add extra module outside the ring
aff_mat = np.eye(4)
aff_mat[:-1, :-1] = Rotation.from_euler(
    'xyz',
    [0, np.pi / 2, 0],
).as_matrix()
aff_mat[0, -1] = -0.75 * num_rings * lor_spacing[1]

mods.append(
    pps.RectangularPETScannerModule((5, 5),
                                    lor_spacing,
                                    ax0=1,
                                    ax1=0,
                                    affine_transformation_matrix=aff_mat))

#------------
mods = tuple(mods)

scanner = pps.ModularizedPETScanner(mods)

# disable coincidences between modules 2 and 1
scanner.set_module_coincidence(1, 2, False)

print(scanner.lor_endpoints[scanner.linear_lor_endpoint_index(
    np.array([1, 4, 3, 7]), np.array([2, 1, 4, 7])), :])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax, show_linear_index=False)
scanner.show_all_lors_for_endpoint(ax, 2, 10)
fig.tight_layout()
fig.show()
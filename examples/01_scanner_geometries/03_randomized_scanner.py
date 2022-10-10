import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pyparallelproj.scanner_modules as pps

np.random.seed(2)

radius = 20.
num_sides = 5
lor_spacing = (5., 4.)
num_lor_endpoints_per_side = 3

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
        pps.RandomizedRectangularPETScannerModule(
            ([(1, 2), (1, 3), (2, 2)][np.random.randint(3)]),
            lor_spacing,
            ax0=1,
            ax1=0,
            affine_transformation_matrix=aff_mat))

#------------
mods = tuple(mods)

scanner = pps.ModularizedPETScannerGeometry(mods)
cd = pps.GenericPETCoincidenceDescriptor(scanner)

cd.setup_lor_lookup_table()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax, show_linear_index=False, annotation_fontsize=6)
cd.show_all_lors(ax)
fig.tight_layout()
fig.show()
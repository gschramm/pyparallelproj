import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scanner_modules as pps

phis = np.linspace(0, 2 * np.pi, 7, endpoint=False)

radius = 30.

mods = []

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

for i, phi in enumerate(phis):
    aff_mat = np.eye(4)
    aff_mat[:-1, :-1] = Rotation.from_euler(
        'xyz',
        [phi, 0, 0],
    ).as_matrix()

    aff_mat[1, -1] = -np.sin(phi) * radius
    aff_mat[2, -1] = np.cos(phi) * radius

    mods.append(
        pps.RectangularPETScannerModule((5, 8), (4., 4., 20.),
                                        ax0=1,
                                        ax1=0,
                                        affine_transformation_matrix=aff_mat))

    mods[-1].show_lor_endpoints(ax,
                                annotation_fontsize=6.,
                                annotation_prefix=f'{i},')

fig.tight_layout()
fig.show()
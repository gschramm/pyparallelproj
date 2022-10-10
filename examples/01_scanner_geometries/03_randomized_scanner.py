import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pyparallelproj.scanner_modules as pps

np.random.seed(2)

radius = 20.
num_sides = 3
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

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scanner.show_lor_endpoints(ax, show_linear_index=False, annotation_fontsize=6)
scanner.show_lor_endpoints(ax2, show_linear_index=False, annotation_fontsize=6)

for mod, num_lor_endpoints in enumerate(scanner.num_lor_endpoints_per_module):
    for lor in range(num_lor_endpoints):
        cd.show_all_lors_for_endpoint(ax, mod, lor)

        startpoint = scanner.linear_lor_endpoint_index(mod, lor)
        tmp = cd.get_modules_and_indicies_in_coincidence(mod, lor)

        tmp = tmp[tmp[:, 0] >= mod]

        if mod == 0 and lor == 0:
            lor_start_module_index = np.repeat(np.array([[mod, lor]]),
                                               tmp.shape[0],
                                               axis=0)
            lor_end_module_index = tmp.copy()
        else:
            lor_start_module_index = np.vstack((lor_start_module_index,
                                                np.repeat(np.array([[mod,
                                                                     lor]]),
                                                          tmp.shape[0],
                                                          axis=0)))
            lor_end_module_index = np.vstack((lor_end_module_index, tmp))

p2s = scanner.get_lor_endpoints(lor_end_module_index[:, 0],
                                lor_end_module_index[:, 1])
p1s = scanner.get_lor_endpoints(lor_start_module_index[:, 0],
                                lor_start_module_index[:, 1])

ls = np.hstack([p1s, p2s]).copy()
ls = ls.reshape((-1, 2, 3))
lc = Line3DCollection(ls, linewidth=0.3)
ax2.add_collection(lc)

fig.tight_layout()
fig.show()
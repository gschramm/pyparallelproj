"""Script to show how to define a generic modular scanner consisting of "random" modules
   This should show how to set up a very generic scanner scanner consisting of modules
   of LOR endpoints that can be anywhere in space.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyparallelproj.scannermodules as scannermods
import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences

np.random.seed(1)

#--------------------------------------------------------------------------------
# scanner parameters

radius = 20.
num_sides = 8
lor_spacing = (5., 4.)
num_lor_endpoints_per_side = 3

#--------------------------------------------------------------------------------
# The scanner consists of multiple rectangular modules where the LOR endpoint
# positions are randomized within the rectangle.
# Using an affine transformation matrix, we can position the modules anywhere
# in space. Here we position them along a ring.

phis = 2 * np.pi * np.random.rand(num_sides)
mods = []

for i, phi in enumerate(phis):
    aff_mat = np.eye(4)
    # rotate the module
    aff_mat[:-1, :-1] = Rotation.from_euler(
        'xyz',
        [-phi, 0, 0],
    ).as_matrix()

    # offset the center of the module
    aff_mat[1, -1] = np.sin(phi) * radius
    aff_mat[2, -1] = np.cos(phi) * radius

    mod = scannermods.RandomizedRectangularPETScannerModule(
        ([(1, 2), (1, 3), (2, 2)][np.random.randint(3)]),
        lor_spacing,
        ax0=1,
        ax1=0,
        affine_transformation_matrix=aff_mat)
    mods.append(mod)

    # you can use mod.show_lor_endpoints() to show only one module

#------------
mods = tuple(mods)
scanner = scanners.ModularizedPETScannerGeometry(mods)

#--------------------------------------------------------------------------------
# setup a coincidence descriptor that describes which LOR endpoints are connected
# here the extra module breaks all symmetries which means we have to stick to
# the generic coincidence descriptor (which is slow for big scanners)
cd = coincidences.GenericPETCoincidenceDescriptor(scanner)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scanner.show_lor_endpoints(ax, show_linear_index=False, annotation_fontsize=6)
cd.show_lors(ax, None)
fig.tight_layout()
fig.show()
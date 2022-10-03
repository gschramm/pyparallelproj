import setuptools
import os
from glob import glob

setuptools.setup(
    name="pyparallelproj",
    use_scm_version={'fallback_version': 'unkown'},
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    author="Georg Schramm,",
    author_email="georg.schramm@kuleuven.be",
    description=
    "python bindings for parallelproj CUDA and OPENMP PET projectors",
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pyparallelproj",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.18', 'matplotlib>=3.2.1', 'numba>=0.49', 'scipy>=1.2',
        'attrs>=22.0'
    ],
    include_package_data=True)

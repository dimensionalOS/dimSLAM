# Copyright 2026 Dimensional Inc.
# SPDX-License-Identifier: Apache-2.0

import sys as _sys

if _sys.platform == "linux":
    # The bundled libcuvslam.so links CUDA libraries that ship in NVIDIA's wheels
    # (declared as dependencies) rather than in this wheel. The dynamic loader will
    # not search site-packages, so load them by SONAME into the global namespace
    # before _native pulls in libcuvslam. Missing ones are skipped: on machines
    # with a system CUDA install ldconfig resolves them, and if a library is truly
    # absent the _native import error names it.
    def _preload_cuda_libs():
        import ctypes
        import importlib.util
        import os

        spec = importlib.util.find_spec("nvidia")
        lib_dirs = [
            os.path.join(root, "cu13", "lib")
            for root in (spec.submodule_search_locations or [] if spec else [])
        ]
        sonames = [
            "libnvJitLink.so.13",
            "libcudart.so.13",
            "libnvrtc.so.13",
            "libcublasLt.so.13",
            "libcublas.so.13",
            "libcusparse.so.12",
            "libcusolver.so.12",
        ]
        for soname in sonames:
            paths = [os.path.join(d, soname) for d in lib_dirs]
            for path in [p for p in paths if os.path.exists(p)] or [soname]:
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass

    _preload_cuda_libs()
    del _preload_cuda_libs

from dim_odom._native import (
    CameraModel,
    CuvslamOdometry,
    ImageFrame,
    ImuNoiseModel,
    ImuSample,
    OdometryEstimate,
    OdometryFusion,
    PointCloud,
    __version__,
    compose,
    init_logging,
    invert,
)

__all__ = [
    "CameraModel",
    "CuvslamOdometry",
    "ImageFrame",
    "ImuNoiseModel",
    "ImuSample",
    "OdometryEstimate",
    "OdometryFusion",
    "PointCloud",
    "__version__",
    "compose",
    "init_logging",
    "invert",
]

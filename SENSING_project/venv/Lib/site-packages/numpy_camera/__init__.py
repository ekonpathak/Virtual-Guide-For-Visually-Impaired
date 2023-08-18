# The MIT License
#
# Copyright (c) 2020 Kevin Walchko

try:
    from importlib.metadata import version # type: ignore
except ImportError:
    from importlib_metadata import version # type: ignore

try:
    from .pinhole_camera import PinholeCamera
except ImportError:
    pass

from .utils import rgb2gray, bgr2gray, gray2rgb, gray2bgr
__license__ = 'MIT'
__author__ = 'Kevin Walchko'
__version__ = version("numpy_camera")

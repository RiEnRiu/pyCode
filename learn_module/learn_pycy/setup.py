# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# python setup.py build_ext --inplace

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(ext_modules=[Extension("cy_nms",
                            ["cy_nms.pyx"],
                            # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
                            include_dirs = [np.get_include()])],
      cmdclass={'build_ext': build_ext})


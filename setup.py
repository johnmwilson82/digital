from distutils.core import setup, Extension
import numpy

setup(
    name="noddy",
    version="1.0",
    ext_modules=[
        Extension("gf2n",
                  sources=["gf2n.c"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['-std=c99', '-g', '-O0']
                  )
    ]
)

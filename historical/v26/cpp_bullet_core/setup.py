from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

try:
    import pybind11
except Exception as exc:
    raise RuntimeError("pybind11 is required: pip install pybind11") from exc

class BuildExt(build_ext):
    c_opts = {
        'unix': ['-O3', '-std=c++17', '-fvisibility=hidden'],
    }
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()

ext_modules = [
    Extension(
        '_cpp_bullet_full_port_stage06',
        ['src/cpp_line_motion_filter.cpp', 'src/pybind_module.cpp'],
        include_dirs=['include', pybind11.get_include()],
        language='c++',
    )
]

setup(
    name='cpp-bullet-full-port-stage06',
    version='0.0.5',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)

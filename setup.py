#! /usr/bin/env python3

import os
import re
import sys
import platform
import subprocess


import os
import sys
import subprocess
import textwrap
import warnings
import builtins


if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

import versioneer


FULLVERSION = versioneer.get_version()
ISRELEASED = 'dev' not in FULLVERSION
MAJOR, MINOR, MICRO = FULLVERSION.split('.')[:3]
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)






# Python supported version checks. Keep right after stdlib imports to ensure we
# get a sensible error for older Python versions
if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")



from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copymode


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print(self)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # Copy *_test file to tests directory
        # test_bin = os.path.join(self.build_temp, 'python_cpp_example_test')
        # self.copy_test_file(test_bin)

        # Add an empty line for cleaner output
        print()

    # def copy_test_file(self, src_file):
    #     """
    #     Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.
    #     By default, message like `creating directory /path/to/package` and
    #     `copying directory /src/path/to/package -> path/to/package` are displayed on standard output.
    #     Adapted from scikit-build.
    #     """
    #     # Create directory if needed
    #     dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'bin')
    #     if dest_dir != "" and not os.path.exists(dest_dir):
    #         print("creating directory {}".format(dest_dir))
    #         os.makedirs(dest_dir)
    #
    #     # Copy file
    #     dest_file = os.path.join(dest_dir, os.path.basename(src_file))
    #     print("copying {} -> {}".format(src_file, dest_file))
    #     copyfile(src_file, dest_file)
    #     copymode(src_file, dest_file)


cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = CMakeBuild

setup(
    name='shapelets',
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    author='Shapelets',
    author_email='justo.ruiz@shapelets.io',
    url="https://shapelets.io",
    description='Time series bla bla bla',
    long_description='Long time series bla bla bla',
    packages=find_packages(where='modules'),
    package_dir={'': 'modules'},
    ext_modules=[CMakeExtension('pygauss')],
    test_suite='tests',
    zip_safe=False,
    package_data={'shapelets': ['libs/*.*']},
    # package_data={
    #     'khiva': ['py.typed', '*.pyi'],
    # },
    # include_package_data=True,
)

# from pybind11_stubgen import ModuleStubsGenerator
# module = ModuleStubsGenerator("khiva")
# module.parse()
# module.write_setup_py = False
#
# with open(cur_file_path + "/khiva.pyi", "w") as fp:
#     fp.write("#\n# AUTOMATICALLY GENERATED FILE, DO NOT EDIT!\n#\n\n")
#     fp.write("\n".join(module.to_lines()))


# class PostDevelopCommand(develop):
#     """Post-installation for development mode."""
#     def run(self):
#         develop.run(self)
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#
# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         install.run(self)
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#
#     # cmdclass={
#     #              'develop': PostDevelopCommand,
#     #              'install': PostInstallCommand,
#     #          },



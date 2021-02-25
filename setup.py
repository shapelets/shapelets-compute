#! /usr/bin/env python3
import os
import sys
import re
import platform
import subprocess
import versioneer

from textwrap import dedent
import warnings
import builtins




def process_version_information(full_version):
    """ Builds a textual version string out of the information provided by versioneer"""
    is_dev = 'dev' not in full_version
    mayor, minor, micro = full_version.split('.')[:3]
    return dict(
        textual='{}.{}.{}{}'.format(mayor, minor, micro, "-dev" if is_dev else ""),
        mayor=mayor,
        minor=minor,
        is_dev=is_dev)


def get_documentation_url(ver_details, doc_root="https:://shapelets.io/doc/"):
    return doc_root + "dev" if ver_details["is_dev"] else "{}.{}".format(ver_details["mayor"], ver_details["minor"])


def create_metadata(full_version, ver_details, doc_url, cmdclass):
    return dict(
        project_urls={
            "Documentation": doc_url
        },
        version=full_version,
        cmdclass=cmdclass,
        python_requires='>=3.7',
    )



from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


##
# Build native libraries using CMAKE
#
# The idea here is to build gauss (mathematical algorithms)
# and pygauss (their Python binding using pybind11) using
# the build process generated from CMAKE.
#
# The output is a single library which will be always placed
# under modules/shapelets/internal folder
#


# class CMakeExtension(Extension):
#     """ It will be used in setup method under ext_modules parameter """
#     def __init__(self, name, sourcedir=''):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """ Actual build action and CMAKE control """

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


def setup_package():
    """ Main entry point that sets everything up"""
    # At least, version 3.7
    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("Python version >= 3.7 required.")

    cmdclass = versioneer.get_cmdclass({"build_ext": CMakeBuild})
    ver_info = versioneer.get_version()
    ver_details = process_version_information(ver_info)

    if sys.version_info >= (3, 10):
        fmt = "Shapelets {} may not yet support Python {}.{}."
        warnings.warn(fmt.format(ver_details["textual"], *sys.version_info[:2]), RuntimeWarning)
        del fmt

    doc_url = get_documentation_url(ver_details)
    meta = create_metadata(ver_info, ver_details, doc_url, cmdclass)

    # Remove MANIFEST to ensure the correct information
    # is generated from MANIFEST.in
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(**meta)


if __name__ == '__main__':
    setup_package()





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

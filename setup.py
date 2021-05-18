#! /usr/bin/env python3
import os
import sys
import re
import platform
import subprocess
import versioneer
from typing import Union, List
from setuptools import Command, setup, Extension, find_packages
from setuptools.command.develop import develop
from distutils.version import LooseVersion
import warnings
import builtins



# Technique from numpy
builtins.__SHAPELETS_SETUP__ = True

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

def check_submodules():
    """ Ensure we have source code for gauss external repos """
    if not os.path.exists('.git'):
        return

    with open('.gitmodules') as f:
        for line in f:
            if 'path' in line:
                p = line.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule {} missing'.format(p))

    proc = subprocess.Popen(['git', 'submodule', 'status'], stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: {}'.format(line))

cmdclass = versioneer.get_cmdclass()

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
class CMakeExtension(Extension):
    """ It will be used in setup method under ext_modules parameter """

    def __init__(self, name, sourcedir='', debug: bool = False, output_dir: str = '', target: Union[List[str], str] = None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target
        self.output_dir = output_dir
        self.debug = debug


class CMakeBuild(cmdclass["build_ext"]):
    """ Actual build action and CMAKE control """

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.9.6':
                raise RuntimeError("CMake >= 3.9.6 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir, ext.output_dir)
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE',
            '-Wno-dev'
        ]

        if 'in_development' in globals():
            cmake_args += [
                '-DCOPY_ALL_FILES=ON'
            ]
        else:
            cmake_args += [
                '-DCOPY_ALL_FILES=OFF'
            ]
        
        cfg = 'Debug' if self.debug else 'Release' # 'RelWithDebInfo'
        build_args = ['--config', cfg]  # cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),                # get existing flags
            self.distribution.get_version()         # plus version
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        if ext.target:
            if isinstance(ext.target, list):
                for target in ext.target:
                    subprocess.check_call(['cmake', '--build', '.', '--target', target] + build_args, cwd=self.build_temp)
            else:
                subprocess.check_call(['cmake', '--build', '.', '--target', ext.target] + build_args, cwd=self.build_temp)
        else:
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class DevelopCommand(develop):
    def run(self):
        global in_development
        in_development = True
        develop.run(self)
        

def create_metadata(full_version, doc_url):
    cmdclass['develop'] = DevelopCommand
    return dict(
        project_urls={
            "Documentation": doc_url
        },
        version=full_version,
        cmdclass=cmdclass,
        packages=find_packages(where='modules'),
        package_dir={'': 'modules'},
        test_suite="pytest",
        ext_modules=[CMakeExtension("pygauss",
                                    debug=False,
                                    output_dir='shapelets/compute',
                                    target=["PyGauss"])],
        python_requires='>=3.7',
        package_data={
            'shapelets': [
                'data/*.txt', 
                'data/*.mat', 
            ],
        },
        include_package_data = True,
    )




def setup_package():
    """ Main entry point that sets everything up """
    # At least, version 3.7
    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("Python version >= 3.7 required.")

    check_submodules()

    cmdclass.update({'build_ext': CMakeBuild})

    ver_info = versioneer.get_version()
    ver_details = process_version_information(ver_info)

    if sys.version_info >= (3, 10):
        fmt = "Shapelets Compute {} may not yet support Python {}.{}."
        warnings.warn(fmt.format(ver_details["textual"], *sys.version_info[:2]), RuntimeWarning)
        del fmt

    doc_url = get_documentation_url(ver_details)
    meta = create_metadata(ver_info, doc_url)

    # Remove MANIFEST to ensure the correct information
    # is generated from MANIFEST.in
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(**meta)


if __name__ == '__main__':
    setup_package()


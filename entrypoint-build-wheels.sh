#!/bin/bash

yum -y install zip
pipx install tox
pipx install twine
pipx install pip

cd io
export LDFLAGS=-L/usr/local/cuda/lib64
tox

/opt/python/cp37-cp37m/bin/python -m pip install cmake auditwheel 

export AUDITWHEEL=`pwd`/auditwheel_patch.py  # the monkey patch script

# for whl in `ls dist/*.whl | grep -v manylinux`; do auditwheel repair $whl -w dist/; rm $whl; done
for whl in `ls dist/*.whl | grep -v manylinux`; do /opt/python/cp37-cp37m/bin/python $AUDITWHEEL repair $whl -w dist/; rm $whl; done


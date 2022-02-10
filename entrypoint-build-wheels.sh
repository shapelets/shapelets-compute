#!/bin/bash

yum -y install zip
pipx install tox
pipx install twine

cd io
export LDFLAGS=-L/usr/local/cuda/lib64
tox

python3.7 -m pip install auditwheel

export AUDITWHEEL=`pwd`/auditwheel_patch.py  # the monkey patch script

# for whl in `ls dist/*.whl | grep -v manylinux`; do auditwheel repair $whl -w dist/; rm $whl; done
for whl in `ls dist/*.whl | grep -v manylinux`; do python3.7 $AUDITWHEEL repair $whl -w dist/; done


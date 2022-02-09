#!/bin/bash

yum -y install zip
pipx install tox
pipx install twine
cd io
export LDFLAGS=-L/usr/local/cuda/lib64
tox
for whl in `ls dist/*.whl | grep -v manylinux`; do auditwheel repair $whl -w dist/; rm $whl; done


#!/bin/bash

pipx install tox 
cd io
tox
for whl in `ls dist/*.whl | grep -v manylinux`; do auditwheel repair $whl -w dist/; rm $whl; done


#!/bin/bash

if ! [ -x "$(command -v curl)" ]; then
  echo 'Error: curl is not installed.' >&2
  exit 1
fi

DIR=${1:-./opt/arrayfire}
VAR=${2:-http://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh}
curl -o linux_install.sh $VAR

[ ! -d "$DIR" ] && mkdir -p $DIR
chmod +x linux_install.sh
bash ./linux_install.sh --prefix=$DIR --skip-license 
rm linux_install.sh




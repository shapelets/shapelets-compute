#!/bin/bash

DIR=${1:-./opt/arrayfire}
ADRS=${2:-http://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh}
FILE=_linux_af.sh
KEEPFILE=true

if ! [ -x "$(command -v curl)" ]; then
  echo 'Error: curl is not installed.' >&2
  exit 1
fi

if [ ! -f "$FILE" ]; then
    KEEPFILE=false
    curl -o $FILE $ADRS
fi

[ -d "$DIR" ] && rm -rf $DIR
mkdir -p $DIR

chmod +x $FILE
bash .$FILE --prefix=$DIR --skip-license 

if [ "$KEEPFILE" = false ] ; then
  rm $FILE
fi



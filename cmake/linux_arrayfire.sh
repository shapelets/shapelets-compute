#!/bin/bash

DIR=${2:-./opt/arrayfire}
ADRS=${3:-http://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh}
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
bash ./$FILE --prefix=$DIR --skip-license 

rm -rf $1
mv ./opt/arrayfire $1
rm -rf ./opt

if [ "$KEEPFILE" = false ] ; then
  echo 'rm $FILE'
fi



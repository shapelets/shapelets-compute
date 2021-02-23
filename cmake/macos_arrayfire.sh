#!/bin/bash

if ! [ -x "$(command -v curl)" ]; then
  echo 'Error: curl is not installed.' >&2
  exit 1
fi

DIR="./.working-temp"
FILE=_mac_af.pkg
VAR=${2:-http://arrayfire.s3.amazonaws.com/3.7.2/ArrayFire-v3.7.2_OSX_x86_64.pkg}  
KEEPFILE=true

if [ ! -f "$FILE" ]; then
    KEEPFILE=false
    curl -o $FILE $VAR
fi

[ -d "$DIR" ] && rm -rf $DIR
mkdir $DIR 

xar -xf $FILE -C $DIR 

for i in `find . -name "Payload"`
do
	cat $i | gzip -d | cpio -id
done

[ -d "$DIR" ] && rm -rf $DIR

if [ "$KEEPFILE" = false ] ; then
  rm $FILE
fi

rm -rf $1
mv ./opt/arrayfire $1
rm -rf ./opt




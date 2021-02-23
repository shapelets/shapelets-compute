#!/bin/bash

if ! [ -x "$(command -v curl)" ]; then
  echo 'Error: curl is not installed.' >&2
  exit 1
fi

VAR=${1:-http://arrayfire.s3.amazonaws.com/3.7.2/ArrayFire-v3.7.2_OSX_x86_64.pkg}  

curl -o af.pkg $VAR

DIR="./.working-temp"

[ -d "$DIR" ] && rm -rf $DIR
mkdir $DIR 

xar -xf af.pkg -C $DIR 

for i in `find . -name "Payload"`
do
	echo $i
	cat $i | gzip -d | cpio -id
done

[ -d "$DIR" ] && rm -rf $DIR
rm af.pkg



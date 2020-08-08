#!/bin/sh

rm ./build -r
mkdir -p ./build/
cd ./build/
cmake ../src
make

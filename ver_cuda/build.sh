#!/bin/sh

rm -rf ./build/
cp src build -r
cd ./build
nvcc ROFtv.cu ROF_tv_VIDEO.cu `pkg-config --libs --cflags opencv` -o TV_VIDEO.out


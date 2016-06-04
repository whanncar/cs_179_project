@echo off
nvcc -O3 main.cc convolve.cu -o convolve -arch=sm_20
IF EXIST convolve.lib. (
    del convolve.lib
)
IF EXIST convolve.exp. (
    del convolve.exp
)

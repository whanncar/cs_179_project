@echo off
nvcc -O3 multShare.cu -o multShare -arch=sm_20

IF EXIST multShare.exp. (
    del multShare.exp
)

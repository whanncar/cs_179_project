@echo off
nvcc -O3 src/multShare.cu  src/utils.cu src/neural_net.cu src/fileio.cu src/main.cu -o run_neural_net -arch=sm_20

IF EXIST run_neural_net.lib. (
    del run_neural_net.lib
)
IF EXIST run_neural_net.exp. (
    del run_neural_net.exp
)
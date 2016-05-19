@echo off
nvcc -O3 NeuralNetwork1.cc NeuralNetwork_cuda.cu -o NeuralNet -arch=sm_20
IF EXIST NeuralNetwork1.lib. (
    del NeuralNetwork1.lib
)
IF EXIST NeuralNetwork1.exp. (
    del NeuralNetwork1.exp
)

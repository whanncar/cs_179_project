@echo off
nvcc -O3 NeuralNetwork2.cc NeuralNetwork2.cu -o NeuralNet2 -arch=sm_20
IF EXIST NeuralNetwork2.lib. (
    del NeuralNetwork2.lib
)
IF EXIST NeuralNetwork2.exp. (
    del NeuralNetwork2.exp
)

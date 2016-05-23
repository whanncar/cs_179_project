/* CPU NEURAL NETWORK, written by
 * Andrew and Wade, 2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdio>
#include <cuda_runtime.h>
#include <time.h>
#include <cassert>


#ifndef CPU_NEURAL_NET_CUH
#define CPU_NEURAL_NET_CUH


struct Array{
   int  r; //n_neurons;
   int  c; //m_inputs;
   float * w;
};


#define IMG_LEN (784)
#define NUM_LAYERS 3

const char* getfield(char* line, int num);

void getDataFromFile(char * inputFile, float * labels, float * data, int numRows, int numCols);

void createTrainLabels(float *trainLabels, float * labels, int numRows, int numClasses);

void multiplyArrays(float * C, float * A, float * B, int r1, int c1, int r2, int c2);

void sigmodForward(float * output, float * input, int length);

float randomFloat();

void initializeWeights(float * weights, int length);

void findMax(float * labelEst, float *  Xout, int numTestSamples, int numOutputNeurons);

float calcLoss(float * labels, float * labelsEst, int numTestSamples);

void calc_dl(float * dl, float * Xl, float * Y, int xl_r, int xl_c, int y_r, int y_c);

void elementWiseMultiply(float * C, float * A, float * B, int r, int c);

void oneMinusArray(float * C, float * A, int r, int c);

void subtractArrays(float * C, float * A, float * B, int r, int c);

void transpose(float * C, float * A, int r, int c);

void scalarMultiply(float * C, float * A, float num, int r, int c);

void initialize_X(struct X, int neuronsPerLayer, int numTestSamples);

void initialize_W(struct W, int neuronsPerLayer, int numTestSamples);

void initialize_layerSum(struct layerSum, int neuronsPerLayer, int numTestSamples);

void initialize_dl(struct dl, int neuronsPerLayer, int numTestSamples);

#endif // CPU_NEURAL_NET_CUH

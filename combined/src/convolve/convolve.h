#ifndef CONVOLVE_H
#define CONVOLVE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../utils/utils.h"

#define NUM_KERNELS 10
#define IMAGE_DIM 28


float pyramid_product(data_matrix *input,
                      data_matrix *kernel,
                      int x, int y);

void stackImages(data_matrix *inputData, data_matrix *kernels, 
                 data_matrix *referenceLabels, int numKernels,
                 int numSamples, int imageDim, int maxPixelVal);

void getDataFromFile(char *inputFile, float *labels,
                     float *data, int numRows, int numCols);

void initializeKernels(data_matrix *kernelsArray,
                       int numKernels, int imageDim);

void convolve(data_matrix *input, data_matrix *kernel,
              data_matrix *output);

const char *getfield(char *line, int num);

data_matrix *convolutional_preprocessing(int numSamples,
                                         int sampleLength, 
                                         char *filepath);

#endif /* CONVOLVE_H */

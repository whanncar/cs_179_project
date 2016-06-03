
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil.h>

#define NUM_KERNELS 10

void initializeKernel(Matrix * inputData, Matrix * kernel, Matrix * labels, int NumericVal);

void initializeAllKernels(Matrix * inputData, Matrix * kernel, Matrix * labels);

void convolveImage(Matrix *input, Matrix *kernel, Matrix *output, Matrix forgot);

const int KERNEL_SIZE = KERNEL_W * sizeof(float);
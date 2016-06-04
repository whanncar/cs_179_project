
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
int width;
int height;
float* elements;
int stride;
} Matrix;


#define NUM_KERNELS 10
#define IMAGE_DIM 28

#define KERNEL_RADIUS 16
#define KERNEL_W 	(2 * KERNEL_RADIUS + 1)

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48


//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

const int KERNEL_SIZE = KERNEL_W * sizeof(float);

//void initializeKernel(Matrix * inputData, Matrix * kernel, Matrix * labels, int NumericVal);

//void initializeAllKernels(Matrix * inputData, Matrix * kernel, Matrix * labels);

float otherproduct(Matrix *input, Matrix *kernel, int x, int y);

void stackImages(Matrix *inputData, Matrix * kernels, Matrix *referenceLabels, int numKernels, int numSamples, int imageDim, int maxPixelVal);

void getDataFromFile(char * inputFile, float * labels, float * data, int numRows, int numCols);

void initializeKernels(Matrix *kernelsArray, int numKernels, int imageDim);

void addpixelApron(Matrix *input, Matrix *paddedInput, int imageSize, int paddedArraySize);

void printImage(Matrix * input);

void CPU_convolution(Matrix *input, Matrix *kernel, Matrix *output, int currentSample);

float calculate_overlap_product(Matrix *input, Matrix *kernel, int x, int y);

void convolve(Matrix *input, Matrix *kernel, Matrix *output);

const char* getfield(char* line, int num);

//const int KERNEL_SIZE = KERNEL_W * sizeof(float);

//void convolveImage(Matrix *input, Matrix *kernel, Matrix *output);



#define UNROLL_INNER
//#include <convolutionSeparable_kernel.cu>


////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Image width should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.



bool loadRawImage(char* filename, int w, int h, float* r, float* g, float* b);

bool writeRawImage(char* filename, int w, int h, float* r, float* g, float* b);

int iDivUp(int a, int b);

int iDivDown(int a, int b);

int iAlignUp(int a, int b);

int iAlignDown(int a, int b);

void convolveImage(Matrix *input, Matrix *kernel, Matrix *output);



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <cutil.h>

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
//Global macro, controlling innermost convolution loop unrolling
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


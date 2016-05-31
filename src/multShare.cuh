/*
* multShare.h
*
* Robert Hochberg
* January 24, 2012
*
* Based nearly entirely on the code from the CUDA C Programming Guide
*/

#ifndef MULTSHARE
#define MULTSHARE

#include "utils.cuh"
#include <stdio.h>
//#include <random>
//#include <curand.h>


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
int width;
int height;
float* elements;
int stride;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

//__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C);

void gpu_calculate_matrix_times_matrix(data_matrix *a, data_matrix *b, data_matrix *c);

#endif
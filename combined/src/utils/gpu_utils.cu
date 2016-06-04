extern "C" {
#include "gpu_utils_cuda.h"
#include <math.h>
}

/* Kernels */


__global__
void shmemTransposeKernel(const float *input, float *output,
                          int num_rows, int num_cols) {


    __shared__ float in_data[32][33];
    __shared__ float out_data[32][33];

    int i, j;

    i = 32 * blockIdx.x + threadIdx.x;
    j = 32 * blockIdx.y + threadIdx.y;

    if (i < num_rows && j < num_cols) {
        in_data[threadIdx.x][threadIdx.y] = input[i * num_cols + j];
    }
    else {
        in_data[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    out_data[threadIdx.y][threadIdx.x] = in_data[threadIdx.x][threadIdx.y];

    __syncthreads();

    i = 32 * blockIdx.y + threadIdx.x;
    j = 32 * blockIdx.x + threadIdx.y;

    if (i < num_cols && j < num_rows) {
        output[i * num_rows + j] = out_data[threadIdx.x][threadIdx.y];
    }

}






__global__
void linCombOfVectors(float a, float *v1, float b, float *v2,
                      int length, float *v_result) {

    float v1_val;
    float v2_val;
    float result;

    int index;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        v1_val = v1[index];
        v2_val = v2[index];

        result = a * v1_val + b * v2_val;

        v_result[index] = result;

    }

}



__global__
void addConstantToVector(float c, float *v1, int length, float *v_result) {

    float v1_val;
    float result;

    int index;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        v1_val = v1[index];

        result = c + v1_val;

        v_result[index] = result;

    }

}



__global__
void multVectsCompwise(float *v1, float *v2, int length, float *v_result) {

    float v1_val;
    float v2_val;
    float result;

    int index;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        v1_val = v1[index];
        v2_val = v2[index];

        result = v1_val * v2_val;

        v_result[index] = result;

    }

}



__global__
void calcVectsSquareDiff(float *v1, float *v2, int length, float *v_result) {

    float v1_val;
    float v2_val;
    float diff;

    int index;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        v1_val = v1[index];
        v2_val = v2[index];

        diff = v1_val - v2_val;

        v_result[index] = diff * diff;

    }

}



__global__
void sumVectorEntries(float *v1, int length, float *result) {

/*

    extern __shared__ float vals[];
*/
    int index;
/*
    int offset_start, offset;

    vals[threadIdx.x] = 0;

*/
    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {
/*
        vals[threadIdx.x] += v1[index];
*/

        atomicAdd(result, v1[index]);
    }

/*
    offset_start = 1;

    while (offset_start < length) {
        offset_start *= 2;
    }

    offset_start /= 2;

    for (offset = offset_start; offset >= 1; offset /= 2) {

        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
            vals[threadIdx.x] += vals[threadIdx.x + offset];
        }

        __syncthreads();

    }

    if (threadIdx.x == 0) {
        atomicAdd(result, vals[threadIdx.x]);
    }
*/



}


__global__
void applySigmoidToVector(float *v1, int length, float *result) {

    int index;
    float input;
    float output;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        input = v1[index];

        output = 1 / (1 + expf(-input));

        result[index] = output; 

    }
}


__global__
void matrixMultiply(float *m1, float *m2, int m1_rows, int m1_cols, int m2_cols, float *result) {


    __shared__ float m1_sub[32][33];
    __shared__ float m2_sub[32][33];
    __shared__ float res_sub[32][33];


    int row, col, k, l;


    /* Initialize the result to 0 */
    res_sub[threadIdx.x][threadIdx.y] = 0;


    for (k = 0; k < m1_cols / 32 + 1; k++) {

        /* Obtain the submatrices */

        row = blockIdx.x * 32 + threadIdx.x;
        col = k * 32 + threadIdx.y;

        if ((row < m1_rows) && (col < m1_cols)) {
            m1_sub[threadIdx.x][threadIdx.y] = m1[row * m1_cols + col];
        }
        else {
            m1_sub[threadIdx.x][threadIdx.y] = 0;
        }

        row = k * 32 + threadIdx.x;
        col = blockIdx.y * 32 + threadIdx.y;

        if ((row < m1_cols) && (col < m2_cols)) {
            m2_sub[threadIdx.x][threadIdx.y] = m2[row * m2_cols + col];
        }
        else {
            m2_sub[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

        /* Multiply the submatrices */

        for (l = 0; l < 32; l++) {
            res_sub[threadIdx.x][threadIdx.y] += m1_sub[threadIdx.x][l] * m2_sub[l][threadIdx.y];
        }

        __syncthreads();

    }

    /* Store the result */

    row = blockIdx.x * 32 + threadIdx.x;
    col = blockIdx.y * 32 + threadIdx.y;

    if ((row < m1_rows) && (col < m2_cols)) {
        result[row * m2_cols + col] = res_sub[threadIdx.x][threadIdx.y];
    }

}




/* Calls */
extern "C"

void callMatrixTranspose(float *d_input,
                         float *d_output,
                         int num_rows,
                         int num_cols)
{
    dim3 blockSize(32, 32);
    dim3 gridSize(num_rows / 32 + 1, num_cols / 32 + 1);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, num_rows, num_cols);
}
 





extern "C"

void callLinCombOfVectors(float a, float *v1, float b, float *v2,
                          int length, float *v_result) {

    int threadsPerBlock;
    int blocks;

    threadsPerBlock = 512;

    blocks = length / threadsPerBlock;

    if (length % threadsPerBlock) {
        blocks++;
    }

    if (blocks > 20000) {
        blocks = 20000;
    }

    linCombOfVectors<<<blocks, threadsPerBlock>>>(a, v1, b, v2, length, v_result);

}


extern "C"

void callAddConstantToVector(float a, float *v1, int length, float *v_result) {

    int threadsPerBlock;
    int blocks;

    threadsPerBlock = 512;

    blocks = length / threadsPerBlock;

    if (length % threadsPerBlock) {
        blocks++;
    }

    if (blocks > 20000) {
        blocks = 20000;
    }

    addConstantToVector<<<blocks, threadsPerBlock>>>(a, v1, length, v_result);

}

extern "C"

void callMultVectsCompwise(float *v1, float *v2, int length, float *v_result) {

    int threadsPerBlock;
    int blocks;

    threadsPerBlock = 512;

    blocks = length / threadsPerBlock;

    if (length % threadsPerBlock) {
        blocks++;
    }

    if (blocks > 20000) {
        blocks = 20000;
    }

    multVectsCompwise<<<blocks, threadsPerBlock>>>(v1, v2, length, v_result);

}

extern "C"

float callCalcVectDist(float *v1, float *v2, int length) {

    int threadsPerBlock;
    int blocks;
    float *sq_diff;
    float result;
    float *result_dev;

    threadsPerBlock = 512;

    blocks = length / threadsPerBlock;

    if (length % threadsPerBlock) {
        blocks++;
    }

    if (blocks > 20000) {
        blocks = 20000;
    }


    cudaMalloc((void **) &sq_diff, length * sizeof(float));

    calcVectsSquareDiff<<<blocks, threadsPerBlock>>>(v1, v2, length, sq_diff);


    cudaMalloc((void **) &result_dev, sizeof(float));

    cudaMemset(result_dev, 0, sizeof(float));

    sumVectorEntries<<<blocks, threadsPerBlock, threadsPerBlock>>>(sq_diff, length, result_dev);

    cudaMemcpy(&result, result_dev, sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(sq_diff);
    cudaFree(result_dev);

    return result;

}

extern "C"
void callApplySigmoidToVector(float *v1, int length, float *result) {

    int threadsPerBlock;
    int blocks;

    threadsPerBlock = 512;

    blocks = length / threadsPerBlock;

    if (length % threadsPerBlock) {
        blocks++;
    }

    if (blocks > 20000) {
        blocks = 20000;
    }

    applySigmoidToVector<<<blocks, threadsPerBlock>>>(v1, length, result);

}

extern "C"
void callMatrixMultiply(float *m1, float *m2, int m1_rows,
                        int m1_cols, int m2_cols, float *result) {

    dim3 blockSize(32, 32);
    dim3 gridSize(m1_rows / 32 + 1, m2_cols / 32 + 1);   

    matrixMultiply<<<gridSize, blockSize>>>(m1, m2, m1_rows, m1_cols, m2_cols, result);

}




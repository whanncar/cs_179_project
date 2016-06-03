#include <math.h>
#include "gpu_utils.cuh"


/* Kernels */


__global__
void shmemTransposeKernel(const float *input, float *output,
                          int num_rows, int num_cols) {

    __shared__ float in_data[65*64];
    __shared__ float out_data[65*64];


    int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;

    int i_data = threadIdx.x;
    int j_data = 4 * threadIdx.y;
    int offset_i_data = i_data + i_data / 32;
    int offset_j_data = j_data + j_data / 32;
    int k;

    for (k = 0; k < 4; k++) {
        if ((i < num_rows) && (j + k < num_cols)) {
            in_data[offset_i_data + 65 * (j_data + k)] = input[i + n * (j + k)];
        }
    }
    __syncthreads();

    for (k = 0; k < 4; k++)
        out_data[offset_j_data + k + 65 * (i_data)]
                = in_data[offset_i_data + 65 * (j_data + k)];

    __syncthreads();

    i = threadIdx.x + 64 * blockIdx.y;
    j = 4 * threadIdx.y + 64 * blockIdx.x;

    for (k = 0; k < 4; k++) {
        if ((i < num_cols) && (j + k < num_rows)) {
            output[i + n * (j + k)] = out_data[offset_i_data + 65 * (j_data + k)];
        }
    }
}

/* Make sure this ^ works */



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
void calcVectsSquareDiff(float *v1, float *v2, int length; float *v_result) {

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

    extern __shared__ float vals[];

    int index;

    int offset_start, offset;

    vals[threadIdx.x] = 0;


    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < length;
         index += gridDim.x * blockDim.x) {

        vals[threadIdx.x] += v1[index];

    }


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
        atomicAdd(vals[threadIdx.x], result);
    }

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


    __shared__ float m1_sub[64][64];
    __shared__ float m2_sub[64][64];
    __shared__ float res_sub[64][64];


    int row, col, k, l;

    /* Initialize the result to 0 */
    res_sub[threadIdx.x][threadIdx.y] = 0;

    for (k = 0; k < m1_cols / 64 + 1; k++) {

        /* Obtain the submatrices */

        row = blockIdx.x * 64 + threadIdx.x;
        col = k * 64 + threadIdx.y;

        if ((row < m1_rows) && (col < m1_cols)) {
            m1_sub[threadIdx.x][threadIdx.y] = m1[row * m1_cols + col];
        }
        else {
            m1_sub[threadIdx.x][threadIdx.y] = 0;
        }

        row = k * 64 + threadIdx.x;
        col = blockIdx.y * 64 + threadIdx.y;

        if ((row < m1_cols) && (col < m2_cols)) {
            m2_sub[threadIdx.x][threadIdx.y] = m2[row * m2_cols + col];
        }
        else {
            m2_sub[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

        /* Multiply the submatrices */

        for (l = 0; l < 64; l++) {
            res_sub[threadIdx.x][threadIdx.y] += m1_sub[threadIdx.x][l] * m2_sub[l][threadIdx.y];
        }

    }

    /* Store the result */

    row = blockIdx.x * 64 + threadIdx.x;
    col = blockIdx.y * 64 + threadIdx.y;

    if ((row < m1_rows) && (col < m2_cols)) {
        result[row * m2_cols + col] = res_sub[threadIdx.x][threadIdx.y];
    }

}




/* Calls */


void callMatrixTranspose(float *d_input,
                         float *d_output,
                         int num_rows,
                         int num_cols)
{
    dim3 blockSize(64, 16);
    dim3 gridSize(num_rows / 64 + 1, num_cols / 64 + 1);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, num_rows, num_cols);
}
 







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

    calcVectsSquareDiff<<<blocks, threadsPerBlock>>>(v1, v2, int length, sq_diff);


    cudaMalloc((void **) &result_dev, sizeof(float));

    cudaMemset(result_dev, 0, sizeof(float));

    sumVectorEntries<<<blocks, threadsPerBlock, threadsPerBlock>>>(sq_diff, int length, result_dev);

    cudaMemcpy(&result, result_dev, sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(sq_diff);
    cudaFree(result_dev);

    return result;

}



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


void callMatrixMultiply(float *m1, float *m2, int m1_rows,
                        int m1_cols, int m2_cols, float *result) {

    dim3 blockSize(64, 64);
    dim3 gridSize(m1_rows / 64 + 1, m2_cols / 64 + 1);   

    matrixMultiply<<<gridSize, blockSize>>>(m1, m2, m1_rows, m1_cols, m2_cols, result);

}

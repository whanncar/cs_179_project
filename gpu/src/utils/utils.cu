








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


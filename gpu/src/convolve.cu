
#include <convolve.cuh>

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}

// Convert float r,g,b to int type
__device__ int rgbToint(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

	////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
                                                float *d_Result,
                                                float *d_Data,
                                                int dataW,
                                                int dataH
                                            )
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[ TILE_H * (TILE_W + KERNEL_RADIUS * 2) ];

    // global mem address of this thread
    const int gLoc = threadIdx.x + 
                            IMUL(blockIdx.x, blockDim.x) +
                            IMUL(threadIdx.y, dataW) +
                            IMUL(blockIdx.y, blockDim.y) * dataW;


    int x;	// image based coordinate

    // original image based coordinate
    const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);

    // case1: left
    x = x0 - KERNEL_RADIUS;
    if ( x < 0 )
        data[threadIdx.x + shift] = 0;
    else 
        data[threadIdx.x + shift] = d_Data[ gLoc - KERNEL_RADIUS];

    // case2: right
    x = x0 + KERNEL_RADIUS;
    if ( x > dataW-1 )
        data[threadIdx.x + blockDim.x + shift] = 0;
    else 
        data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS];

    __syncthreads();

    // convolution
    float sum = 0;
    x = KERNEL_RADIUS + threadIdx.x;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
        sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

    d_Result[gLoc] = sum;

}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColGPU(
                                    float *d_Result,
                                    float *d_Data,
                                    int dataW,
                                    int dataH
                                 )
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];

    // global mem address of this thread
    const int gLoc = threadIdx.x + 
                        IMUL(blockIdx.x, blockDim.x) +
                        IMUL(threadIdx.y, dataW) +
                        IMUL(blockIdx.y, blockDim.y) * dataW;

    int y;	// image based coordinate

    // original image based coordinate
    const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    const int shift = threadIdx.y * (TILE_W);

    // case1: upper
    y = y0 - KERNEL_RADIUS;
    if ( y < 0 )
        data[threadIdx.x + shift] = 0;
    else 
        data[threadIdx.x + shift] = d_Data[ gLoc - IMUL(dataW, KERNEL_RADIUS)];

    // case2: lower
    y = y0 + KERNEL_RADIUS;
    const int shift1 = shift + IMUL(blockDim.y, TILE_W);
    if ( y > dataH-1 )
        data[threadIdx.x + shift1] = 0;
    else 
        data[threadIdx.x + shift1] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];

    __syncthreads();

    // convolution
    float sum = 0;
    for (int i = 0; i <= KERNEL_RADIUS*2; i++)
        sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

    d_Result[gLoc] = sum;

}



//input data is dimensions numRows = numSamples = inputData.height
//numCols = numPixels per image = inputData.width
void initializeKernel(Matrix * inputData, Matrix * kernel, Matrix * labels, int NumericVal)
{
	int count = 0;
	float sum = 0;

	//Zero all elements of kernel
	for(int  j = 0; j < inputData.width; j++)  //For each pixel in the image
	{
		kernel.elements[j] = 0;
	}

	for(int  j = 0; j < inputData.width)  //For each pixel in the image
	{
		sum = 0;
		//Add all the images together to great kernels 
		for(int i = 0; i < inputData.height; i++)  //for each image in the dataset
		{
			if(labels.elements[i] == NumericVal)
			{
				sum = sum + inputData.elements[i];
				count ++;
			}
		}
		kernel.elements[j] = sum;
	}
}

void initializeAllKernels(Matrix * inputData, Matrix * kernel, Matrix * labels)
{
	for(int i = 0; i < NUM_KERNELS; i++)
	{
		initializeKernel(inputData, kernel[i],labels, int NumericVal);
	}
}


void convolveImage(Matrix *input, Matrix *kernel, Matrix *output, Matrix forgot)

    float *d_DataA, *d_DataB;
    int data_size;

    int pixelApron = 14;
	int dw = 28 + pixelApron;
    int dh = 28 + pixelApron;
    
    int data_size = dw*dh;
    
    kernel_size = 28*28;

    printf("Initializing data...\n");
   
    CUDA_SAFE_CALL( cudaMalloc( (void **)&d_DataA, data_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&d_DataB, data_size) );
    cudaMemset()

    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernel, kernel.elements, kernel_size) );

	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, input->elements, data_size, cudaMemcpyHostToDevice) );

    dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
    dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);	// 16 128 8
    dim3 threadBlockColumns(COLUMN_TILE_W, 8);

	printf("=============================================================\n");
	printf(" CUDA Convolution: Image Resolution %i x %i\n", dw, dh);
	printf("=============================================================\n");

	// red channel
	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataR, data_size, cudaMemcpyHostToDevice) );
	
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	convolutionRowGPU<<<blockGridRows, threadBlockRows>>>(
        d_DataB,
        d_DataA,
        dw,
        dh
	);
	CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");

	convolutionColumnGPU<<<blockGridColumns, threadBlockColumns>>>(
        d_DataA,
        d_DataB,
        dw,
        dh,
        COLUMN_TILE_W * threadBlockColumns.y,
        dw * threadBlockColumns.y
	);
	CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// read back GPU result
	CUDA_SAFE_CALL( cudaMemcpy(output->elements, d_DataA, data_size, cudaMemcpyDeviceToHost) );

}

    

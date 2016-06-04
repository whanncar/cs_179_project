/*
 * Original source from nvidia cuda SDK 2.0
 * Modified by S. James Lee (sjames@evl.uic.edi)
 * 2008.12.05
 */

/*
 * This sample implements a separable convolution filter 
 * of a 2D signal with a gaussian kernel.
 */

/*
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil.h>*/

#include "convolveSeparable_header.cuh"




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



void convolveImage(Matrix *input, Matrix *kernel, Matrix *output)

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


    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernel, kernel.elements, kernel_size) );

    CUDA_SAFE_CALL( cudaMemcpy(d_DataA, input->elements, data_size, cudaMemcpyHostToDevice) );

    dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
    dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);   // 16 128 8
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

//Carry out dummy calculations before main computation loop
//in order to "warm up" the hardhare/driver
//#define WARMUP

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
/*
int main(int argc, char **argv){
    
    float *h_Kernel;
	float *h_DataR, *h_DataG, *h_DataB, *h_ResultR, *h_ResultG, *h_ResultB;
    float *d_DataA, *d_DataB;

    double gpuTime, runTime, singleRunTime;

    int i, dw, dh, data_size, repeat;
	dw = dh = 1024;
	repeat = 10;
	
    unsigned int hTimer;

    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));

	// check arg: image resolution
	char *iFilename = "../../../../hubble/hubble1kby1k.raw";
	char *oFilename = "hubble1kby1k_out.raw";
	cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &dw);
	switch (dw)
    {
    case 1024:
    	iFilename = "../../../../hubble/hubble1kby1k.raw";
    	oFilename = "hubble1kby1k_out.raw";
    	dh = dw;
    	break;
    case 2048:
    	iFilename = "../../../../hubble/hubble2kby2k.raw";
    	oFilename = "hubble2kby2k_out.raw";
    	dh = dw;
    	break;
    case 4096:
    	iFilename = "../../../../hubble/hubble4kby4k.raw";
    	oFilename = "hubble4kby4k_out.raw";
    	dh = dw;
    	break;
    default:
    	dh = dw = 1024;
    	printf("use image resoluiton one of 1024, 2048, 4096...\n");
    	printf("will use 1024x1024 as default resolution this time.\n");
    	break;
    }	
	data_size = dw * dh * sizeof(int);
	
	// total iteration number for mean value
	cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &repeat);
	
    printf("Initializing data...\n");
    h_Kernel    = (float *)malloc(KERNEL_SIZE);
    
    h_DataR		= (float *)malloc(data_size);
    h_DataG		= (float *)malloc(data_size);
    h_DataB		= (float *)malloc(data_size);
    h_ResultR	= (float *)malloc(data_size);
    h_ResultG	= (float *)malloc(data_size);
    h_ResultB	= (float *)malloc(data_size);
    
    CUDA_SAFE_CALL( cudaMalloc( (void **)&d_DataA, data_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&d_DataB, data_size) );

	// initialize kernel
    float kernelSum = 0;
    for(i = 0; i < KERNEL_W; i++){
        float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
        h_Kernel[i] = expf(- dist * dist / 2);
        kernelSum += h_Kernel[i];
    }
    for(i = 0; i < KERNEL_W; i++)
        h_Kernel[i] /= kernelSum;

    if (!loadRawImage(iFilename, dw, dh, h_DataR, h_DataG, h_DataB) )
    {
    	printf("File not found. random image generator will be used...\n");
        	
      	srand(2007);
       	for(i = 0; i < dw * dh; i++)
       	{
       	    h_DataR[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       	    h_DataG[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       	    h_DataB[i] = (float)rand() / (float)RAND_MAX * 255.0f;
       	}       	
    }
    
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE) );
    CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataR, data_size, cudaMemcpyHostToDevice) );

    dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
    dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);	// 16 128 8
    dim3 threadBlockColumns(COLUMN_TILE_W, 8);


#ifdef WARMUP
    printf("Warm up ");

	for (i=0; i<1; i++)
	{    
    	// red channel
    	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataR, data_size, cudaMemcpyHostToDevice) );
    	CUDA_SAFE_CALL( cudaThreadSynchronize() );
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

    	// green channel
    	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataB, data_size, cudaMemcpyHostToDevice) );
    	
    	CUDA_SAFE_CALL( cudaThreadSynchronize() );
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

    	// blue channel
    	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataB, data_size, cudaMemcpyHostToDevice) );
    	
    	CUDA_SAFE_CALL( cudaThreadSynchronize() );
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

    	printf(".");
	}
	printf("\n");
#endif

	printf("=============================================================\n");
	printf(" CUDA Convolution: Image Resolution %i x %i\n", dw, dh);
	printf("=============================================================\n");

	runTime = 0;	
	for (i = 0; i < repeat; i++)
	{
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
    	CUT_SAFE_CALL(cutStopTimer(hTimer));
    	gpuTime = cutGetTimerValue(hTimer);
    	runTime += gpuTime;
    	singleRunTime = gpuTime;
    	//printf("%ith GPU convolution time : %f msec\n", i, gpuTime);

		// read back GPU result
		if (i == (repeat -1))
    		CUDA_SAFE_CALL( cudaMemcpy(h_ResultR, d_DataA, data_size, cudaMemcpyDeviceToHost) );
    
    	// green channel
    	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataG, data_size, cudaMemcpyHostToDevice) );
    	
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
    	CUT_SAFE_CALL(cutStopTimer(hTimer));
    	gpuTime = cutGetTimerValue(hTimer);
    	runTime += gpuTime;
    	singleRunTime += gpuTime;
    	//printf("%ith GPU convolution time : %f msec\n", i, gpuTime);

		// read back GPU result
		if (i == (repeat -1))
    		CUDA_SAFE_CALL( cudaMemcpy(h_ResultG, d_DataA, data_size, cudaMemcpyDeviceToHost) );

    	// blue channel
    	CUDA_SAFE_CALL( cudaMemcpy(d_DataA, h_DataB, data_size, cudaMemcpyHostToDevice) );
    	
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
    	CUT_SAFE_CALL(cutStopTimer(hTimer));
    	gpuTime = cutGetTimerValue(hTimer);
    	runTime += gpuTime;
    	singleRunTime += gpuTime;
    	printf("%ith GPU convolution time : %f msec\n", i, singleRunTime);

		// read back GPU result
		if (i == (repeat -1))
    		CUDA_SAFE_CALL( cudaMemcpy(h_ResultB, d_DataA, data_size, cudaMemcpyDeviceToHost) );
	}

	printf("=============================================================\n");
	printf(" Convolution Time: %f msecs (mean of %i run)\n", runTime/ repeat, repeat);
	printf("=============================================================\n\n");
	
	// write result image
	writeRawImage(oFilename, dw, dh, h_ResultR, h_ResultG, h_ResultB);

    printf("Shutting down...\n");
    CUDA_SAFE_CALL( cudaFree(d_DataB) );
    CUDA_SAFE_CALL( cudaFree(d_DataA) );
    
    free(h_ResultB);
    free(h_ResultG);
    free(h_ResultR);
    free(h_DataB);
    free(h_DataG);
    free(h_DataR);
    free(h_Kernel);

    CUT_SAFE_CALL(cutDeleteTimer(hTimer));
}


void convolveImage(Matrix *input, Matrix *kernel, Matrix *output)

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


    CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernel, kernel.elements, kernel_size) );

    CUDA_SAFE_CALL( cudaMemcpy(d_DataA, input->elements, data_size, cudaMemcpyHostToDevice) );

    dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
    dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);   // 16 128 8
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

}*/
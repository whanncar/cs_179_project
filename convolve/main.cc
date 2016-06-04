    
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "convolve_header.cuh"
#include "convolutionSeparable_header.cuh"



int main(int argc, char **argv) {

    if (argc != 6)
    {
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input training file's name > \n \
        < Num Rows in data > \n \
        < Num Columns in data >\n \
        < threads per block >\n \
        < number of blocks >\n");
        exit(EXIT_FAILURE);
    }

    const int numRows = atoi(argv[2]);
    const int numCols = atoi(argv[3]);
    const unsigned int threadsPerBlock = atoi(argv[4]);
    const unsigned int maxBlocks = atoi(argv[5]);

    Matrix * kernels = (Matrix *) malloc(NUM_KERNELS * sizeof(Matrix));
    Matrix * inputData = (Matrix *) malloc(sizeof(Matrix));
    Matrix * referenceLabels = (Matrix *) malloc(sizeof(Matrix));
    Matrix * output = (Matrix *) malloc(sizeof(Matrix));

    float count = 0;
    int numSamples = 10000;
    float MaxPixVal = 255;

    output->elements = (float *) malloc(IMAGE_DIM*IMAGE_DIM* sizeof(float));
    output->width = IMAGE_DIM;
    output->height = IMAGE_DIM;
    output->stride = IMAGE_DIM;

    referenceLabels->elements = (float *) malloc(numRows* sizeof(float));
    referenceLabels->width = 1;
    referenceLabels->height = numRows;
    referenceLabels->stride = 1;

    inputData->elements = (float *) malloc(numRows*(numCols + 1)* sizeof(float));
    inputData->width = numCols + 1;
    inputData->height = numRows;
    inputData->stride = numCols + 1;

    getDataFromFile(argv[1], referenceLabels->elements, inputData->elements, numRows, numCols);

    initializeKernels(kernels, NUM_KERNELS, IMAGE_DIM);

    stackImages(inputData, kernels, referenceLabels, NUM_KERNELS, numSamples, IMAGE_DIM, MaxPixVal);

    //for(int i = 0; i < NUM_KERNELS; i++)

    /*int paddedArraySize = 64;
    Matrix *padded_input = (Matrix *) malloc(sizeof(Matrix));
    padded_input->elements = (float *) malloc(paddedArraySize*paddedArraySize* sizeof(float));
    padded_input->width = IMAGE_DIM;
    padded_input->height = IMAGE_DIM;
    padded_input->stride = IMAGE_DIM * IMAGE_DIM;*/

    //addpixelApron(inputData, padded_input, IMAGE_DIM, paddedArraySize);
   //
    //convolveImage(padded_input, kernels, output);

    printf("here");
    convolve(inputData, kernels + 7, output);

    //Matrix *image = (Matrix *) malloc(sizeof(Matrix));
    //image->elements = (float *) malloc(IMAGE_DIM*IMAGE_DIM* sizeof(float));

    //int currentSample = 0;

    //CPU_convolution(inputData, kernels, image, currentSample);

    printf("convolveImage output\n");
    printImage(output);


    printf("Input Image\n");
    printImage(inputData);


    printf("Kernels\n");
    printImage(kernels + 7);

    //}
   /* float maxCorr = 0;
    for(int i = 0; i < IMAGE_DIM; i++)
    {
        for(int j = 0; j < IMAGE_DIM; j++)
        {
            if(maxCorr < output->elements[i*IMAGE_DIM + j])
            {
                maxCorr = output->elements[i*IMAGE_DIM + j];
            }
        }
    }
    printf("MAX Corr = %f",maxCorr);*/


    printf("\n");
    printf("Reference Labels\n");
    printImage(referenceLabels);

    printf("sum kernel \n");
    printImage(kernels + 7);

    /*
    for(int i = 0; i < 28; i++)     //for each pixel
    {
        for(int j = 0; j < 28; j++)
        {
            if(kernels[7].elements[i*28+j] <= .1)
            {
                printf(" ");
            }
            else{
                printf(" %1.2f", kernels[7].elements[i*28+j]);
            }
        }
        printf("\n");
    }*/

printf("sum kernel \n");
printImage(kernels + 8);
/*
    for(int i = 0; i < 28; i++)     //for each pixel
    {
        for(int j = 0; j < 28; j++)
        {
            if(kernels[8].elements[i*28+j] <= .1)
            {
                printf(" ");
            }
            else{
                printf(" %1.2f", kernels[8].elements[i*28+j]);
            }
            
        }
        printf("\n");
    }*/

    return 0;
}

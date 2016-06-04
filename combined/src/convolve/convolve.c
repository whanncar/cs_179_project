#include "convolve.h"


void initializeKernels(data_matrix *kernelsArray,
                       int numKernels, int imageDim) {

    int i, j, k;

    for(k = 0; k < numKernels; k++) {

        kernelsArray[k].data      = (float *) malloc(imageDim * imageDim * sizeof(float));
        kernelsArray[k].num_cols  = imageDim;
        kernelsArray[k].num_rows  = imageDim;
        kernelsArray[k].stride    = imageDim;

        for(i = 0; i < imageDim; i++) {
            for(j = 0; j < imageDim; j++) {
                kernelsArray[k].data[i*imageDim + j] = 0;
            }
        }
    }
}


void stackImages(data_matrix *inputData, data_matrix * kernels,
                 data_matrix *referenceLabels, int numKernels,
                 int numSamples, int imageDim, int maxPixelVal) {

    int numericVals, k, i, j;
    float count = 0;

    for (numericVals = 0; numericVals < numKernels; numericVals ++) {

        count = 0;
        /* For images k */
        for (k = 0; k < numSamples; k++) {
            /* If image matches */
            if (referenceLabels->data[k] == numericVals) {
                count = count + 1;
                /* For each pixel */
                for (i = 0; i <  IMAGE_DIM; i++) {
                    for (j = 0; j <  IMAGE_DIM; j++) {
                        kernels[numericVals].data[i* IMAGE_DIM+j] +=
                            inputData->data[k*(IMAGE_DIM* IMAGE_DIM) + i* IMAGE_DIM + j]; 
                    }
                }
            }
        }

        for (i = 0; i < IMAGE_DIM; i++)     //for each pixel normalize the image
        {
            for(j = 0; j < IMAGE_DIM; j++)
            {
                kernels[numericVals].data[i * IMAGE_DIM + j] =
                    kernels[numericVals].data[i * IMAGE_DIM + j]/(count * maxPixelVal); 
            }
        }
    }
}


void getDataFromFile(char * inputFile, float * labels, float * data, int numRows, int numCols)
{
    char  buffer[2048] ;
    char  *record, *line;
    int   i = 0, j = 0;
    int   err  = -1;
    FILE *fstream  = fopen(inputFile,"r");

    if(fstream == NULL)
    {
        printf("File read error on opening\n");
    }

    printf("Begin read training data from File\n");
    while(((line=fgets(buffer,sizeof(buffer),fstream))!=NULL) && (i < numRows))
    {
        record = strtok(line,",");
        j = 0;
        //Read label for each image //label is the first data point in a row
        labels[i] = atoi(record);
        record = strtok(NULL,",");
        //printf("%d %d  ",i,labels[i]);

        while(record != NULL && (j < numCols))  
        {
            data[i*numCols + j] = atoi(record);
            record = strtok(NULL,",");
            j++; 
        }
        //printf("%d %d\n", j-1, data[i*(numCols + 1) + j-1] );
        ++i ;
    }

    err = fclose(fstream);
    if(err != 0)
    {
        printf("Error closing file");
    }
}

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ";");
            tok && *tok;
            tok = strtok(NULL, ";\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}



float pyramid_product(data_matrix *input, data_matrix *kernel, int x, int y) {

    float sum  = 0;
    int i, j;

    for(i = 0; i < kernel->num_rows; i++) //Row
    {
        for(j = 0; j < kernel->num_cols; j++) //cols
        {
            if ((x + i < 0) || (x + i >= IMAGE_DIM) || (y + j < 0) || (y + j >= IMAGE_DIM)) {
                continue;
            }
            sum += input->data[(x + i)*IMAGE_DIM + (y + j)]*kernel->data[i*IMAGE_DIM + j];
        }

    }
    return sum/(IMAGE_DIM*IMAGE_DIM);
}


void convolve(data_matrix *input, data_matrix *kernel, data_matrix *output) {

    int apron;
    int x, y;

    apron = kernel->num_cols / 2;

    for (x = -apron; x < IMAGE_DIM - apron; x++) {
        for (y = -apron; y < IMAGE_DIM - apron; y++) {

            output->data[(apron - y) * output->num_cols + (x + apron)] =
                                            pyramid_product(input, kernel, x, y);

        }
    }
}


void convolve_all(data_matrix *input, data_matrix *kernel,
                  data_matrix *output, int num_samples) {

    int i;

    data_matrix *temp_input;
    data_matrix *temp_output;

    temp_input = (data_matrix *) malloc(sizeof(data_matrix));
    temp_output = (data_matrix *) malloc(sizeof(data_matrix));

    temp_input->num_rows = IMAGE_DIM;
    temp_input->num_cols = IMAGE_DIM;

    temp_output->num_rows = IMAGE_DIM;
    temp_output->num_cols = IMAGE_DIM;

    for (i = 0; i < num_samples; i++) {
        temp_input->data = input->data + i * IMAGE_DIM * IMAGE_DIM;
        temp_output->data = output->data + i * IMAGE_DIM * IMAGE_DIM;
        convolve(temp_input, kernel, temp_output);
    }    
/*
    free(temp_input);
    free(temp_output);
*/
}





data_matrix *convolutional_preprocessing(int numSamples, int sampleLength, char *filepath) {

    data_matrix * kernels = (data_matrix *) malloc(NUM_KERNELS * sizeof(data_matrix));
    data_matrix * inputData = (data_matrix *) malloc(sizeof(data_matrix));
    data_matrix * referenceLabels = (data_matrix *) malloc(sizeof(data_matrix));
    data_matrix * output = (data_matrix *) malloc(sizeof(data_matrix));

    float count = 0;
    float MaxPixVal = 255;

    output->data = (float *) malloc(IMAGE_DIM*IMAGE_DIM*numSamples*sizeof(float));
    output->num_cols = IMAGE_DIM * IMAGE_DIM;
    output->num_rows = numSamples;
    output->stride = IMAGE_DIM;

    referenceLabels->data = (float *) malloc(numSamples * sizeof(float));
    referenceLabels->num_cols = 1;
    referenceLabels->num_rows = numSamples;
    referenceLabels->stride = 1;

    inputData->data = (float *) malloc(numSamples * sampleLength * sizeof(float));
    inputData->num_cols = sampleLength;
    inputData->num_rows = numSamples;
    inputData->stride = sampleLength;

    getDataFromFile(filepath, referenceLabels->data, inputData->data, numSamples, sampleLength);

    initializeKernels(kernels, NUM_KERNELS, IMAGE_DIM);
    printf("stackImages\n");
    stackImages(inputData, kernels, referenceLabels, NUM_KERNELS, numSamples, IMAGE_DIM, MaxPixVal);
    printf("convolve_All\n");
    convolve_all(inputData, kernels, output, numSamples);
    return output;
}

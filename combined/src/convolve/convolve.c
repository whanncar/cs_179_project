#include "convolve.h"


void initializeKernels(data_matrix *kernelsArray,
                       int numKernels, int imageDim) {

    for(int k = 0; k < numKernels; k++) {

        kernelsArray[k].data      = (float *) malloc(imageDim * imageDim * sizeof(float));
        kernelsArray[k].num_cols  = imageDim;
        kernelsArray[k].num_rows  = imageDim;
        kernelsArray[k].stride    = imageDim;

        for(int i = 0; i < imageDim; i++) {
            for(int j = 0; j < imageDim; j++) {
                kernelsArray[k].data[i*imageDim + j] = 0;
            }
        }
    }
}


void stackImages(data_matrix *inputData, data_matrix * kernels,
                 data_matrix *referenceLabels, int numKernels,
                 int numSamples, int imageDim, int maxPixelVal) {

    float count = 0;

    for (int numericVals = 0; numericVals < numKernels; numericVals ++) {

        count = 0;
        /* For images k */
        for (int k = 0; k < numSamples; k++) {
            /* If image matches */
            if (referenceLabels->data[k] == numericVals) {
                count = count + 1;
                /* For each pixel */
                for (int i = 0; i <  IMAGE_DIM; i++) {
                    for (int j = 0; j <  IMAGE_DIM; j++) {
                        kernels[numericVals].data[i* IMAGE_DIM+j] +=
                            inputData->data[k*(IMAGE_DIM* IMAGE_DIM + 1) + i* IMAGE_DIM + j]; 
                    }
                }
            }
        }

        for (int i = 0; i < IMAGE_DIM; i++)     //for each pixel normalize the image
        {
            for(int j = 0; j < IMAGE_DIM; j++)
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
            data[i*(numCols + 1) + j] = atoi(record);
            record = strtok(NULL,",");
            j++;
            if(j == numCols) //if j == numCol -1 (since j++ )
            {
                data[i*(numCols + 1) + j] = 1;  //Add one extra element at the end for a bias
                //printf("  %f", data[i*(numCols + 1) + j]);
            }
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

    for(int i = 0; i < kernel->num_rows; i++) //Row
    {
        for(int j = 0; j < kernel->num_cols; j++) //cols
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

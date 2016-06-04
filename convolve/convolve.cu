
#include "convolve_header.cuh"
//#include "convolveSeparable_header.cuh"


void initializeKernels(Matrix *kernelsArray, int numKernels, int imageDim)
{
    for(int k = 0; k < numKernels; k++)
    {
        kernelsArray[k].elements = (float *) malloc(imageDim * imageDim * sizeof(float));
        kernelsArray[k].width    = imageDim;
        kernelsArray[k].height   = imageDim;
        kernelsArray[k].stride   = imageDim;

        for(int i = 0; i < imageDim; i++)
        {
            for(int j = 0; j < imageDim; j++)
            {
                kernelsArray[k].elements[i*imageDim + j] = 0;
            }
        }
    }
}


void CPU_convolution(Matrix *input, Matrix *kernel, Matrix *output, int currentSample)
{
    int m = input->width;
    int n = input->height;
    if(m != n)
    {
        printf("Error input is not square\n");
    }

    int i, j, ii, jj;
    int K = kernel->width;
    float scale = 1;
    int sum = 0;

    if(K != kernel->height)
    {
        printf("Error Kernel not square\n");
    }

    // in, out are m x n images (integer data)
    // K is the kernel size (KxK) - currently needs to be an odd number, e.g. 3
    // coeffs[K][K] is a 2D array of integer coefficients
    // scale is a scaling factor to normalise the filter gain

    for (i = K / 2; i < m - K / 2; ++i) // iterate through image
    {
      for (j = K / 2; j < n - K / 2; ++j)
      {
        sum = 0; // sum will be the sum of input data * coeff terms

        for (ii = - K / 2; ii <= K / 2; ++ii) // iterate over kernel
        {
          for (jj = - K / 2; jj <= K / 2; ++jj)
          {
            //int data = input[i + ii][j +jj];
            int data = input->elements[currentSample*input->stride + (i + ii)*IMAGE_DIM + (j + jj)];
            int coeff = kernel->elements[(ii + K / 2)*IMAGE_DIM + (jj + K / 2)];

            sum += data * coeff;
          }
        }
        output->elements[i*IMAGE_DIM + j] = sum / scale; // scale sum of convolution products and store in output
      }
    }
}


void printImage(Matrix * input)
{
    for(int i = 0; i < IMAGE_DIM; i++)
    {
        for(int j = 0; j < IMAGE_DIM; j++)
        {
            if(input->elements[i*IMAGE_DIM + j] == 0)
            {
                printf(" "); 
            }
            else
            {
                printf(" %2.1f", input->elements[i*IMAGE_DIM + j] );
            }
        }
        printf("\n");
    }
}

void stackImages(Matrix *inputData, Matrix * kernels, Matrix *referenceLabels, int numKernels, int numSamples, int imageDim, int maxPixelVal)
{
    float count = 0;

    for(int numericVals = 0; numericVals < numKernels; numericVals ++)
    {
        count = 0;
        for(int k = 0; k < numSamples; k++) //for images k
        {
            if(referenceLabels->elements[k] == numericVals)   //if image matches
            {
                count = count + 1;
                for(int i = 0; i <  IMAGE_DIM; i++)     //for each pixel
                {
                    for(int j = 0; j <  IMAGE_DIM; j++)
                    {
                        kernels[numericVals].elements[i* IMAGE_DIM+j] += inputData->elements[k*(IMAGE_DIM* IMAGE_DIM + 1) + i* IMAGE_DIM + j]; 
                    }
                }
            }
        }

        for(int i = 0; i < IMAGE_DIM; i++)     //for each pixel normalize the image
        {
            for(int j = 0; j < IMAGE_DIM; j++)
            {
                kernels[numericVals].elements[i * IMAGE_DIM + j] = kernels[numericVals].elements[i * IMAGE_DIM + j]/(count * maxPixelVal); 
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

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
//#define KERNEL_RADIUS 8
//#define      KERNEL_W (2 * KERNEL_RADIUS + 1)

__device__ __constant__ float d_Kernel[KERNEL_W];

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float *data){

    return
        data[KERNEL_RADIUS - i] * d_Kernel[i]
        + convolutionRow<i - 1>(data);
}

template<> __device__ float convolutionRow<-1>(float *data){
    return 0;
}

template<int i> __device__ float convolutionColumn(float *data){
    return 
        data[(KERNEL_RADIUS - i) * COLUMN_TILE_W] * d_Kernel[i]
        + convolutionColumn<i - 1>(data);
}

template<> __device__ float convolutionColumn<-1>(float *data){
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH )
{
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = IMUL(blockIdx.y, dataW);

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
    //of half-warp size, rowStart + apronStartAligned is also a 
    //multiple of half-warp size, thus having proper alignment 
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] = 
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }


    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tileStart + threadIdx.x;
    
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;

#ifdef UNROLL_INNER
        sum = convolutionRow< 2 * KERNEL_RADIUS >(data + smemPos);
#else
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel[KERNEL_RADIUS - k];
#endif

        d_Result[rowStart + writePos] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride)
{
    //Data cache
    __shared__ float data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
    
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] = 
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data, 
    //loaded by another threads
    __syncthreads();
    
    //Shared and global memory indices for current column
    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;
    
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;

#ifdef UNROLL_INNER
        sum = convolutionColumn<2 * KERNEL_RADIUS>(data + smemPos);
#else
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += 
                data[smemPos + IMUL(k, COLUMN_TILE_W)] *
                d_Kernel[KERNEL_RADIUS - k];
#endif

        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}


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

void addpixelApron(Matrix *input, Matrix *paddedInput, int imageSize, int paddedArraySize)
{

    int padBuffer = (paddedArraySize - imageSize)/2;
    int n,m;

    for(int i = 0; i < paddedArraySize; i++)
    {
        for(int j = 0; j < paddedArraySize; j++)
        {
           paddedInput->elements[i*paddedArraySize + j] = 0;
        }
    }

    for(int i = padBuffer; i < paddedArraySize - paddedArraySize; i++)
    {
        n = i - padBuffer;
        
        for(int j = padBuffer; j < paddedArraySize - paddedArraySize; j++)
        {
            m = j - padBuffer;

            paddedInput->elements[i*paddedArraySize + j] = input->elements[n*imageSize + m];
        }
    }
}

void convolveImage(Matrix *input, Matrix *kernel, Matrix *output)
{
    float *d_DataA, *d_DataB;
    //int data_size;

    //int pixelApron = 14;
    int dw = IMAGE_DIM;
    int dh = IMAGE_DIM;
    
    int data_size = 64*64*sizeof(float); //dw*dh;
    
    //int kernel_size = IMAGE_DIM*IMAGE_DIM;

    printf("Initializing data...\n");
   
    cudaMalloc( (void **)&d_DataA, data_size);
    cudaMalloc( (void **)&d_DataB, data_size);

    cudaMemcpyToSymbol(d_Kernel, kernel->elements, KERNEL_SIZE);

    dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
    dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);   // 16 128 8
    dim3 threadBlockColumns(COLUMN_TILE_W, 8);

    printf("=============================================================\n");
    printf(" CUDA Convolution: Image Resolution %i x %i\n", dw, dh);
    printf("=============================================================\n");

    // red channel
    cudaMemcpy(d_DataA, input->elements, data_size, cudaMemcpyHostToDevice);
    
    cudaThreadSynchronize();


    convolutionRowGPU<<<blockGridRows, threadBlockRows>>>(
        d_DataB,
        d_DataA,
        dw,
        dh
    );


    convolutionColumnGPU<<<blockGridColumns, threadBlockColumns>>>(
        d_DataA,
        d_DataB,
        dw,
        dh,
        COLUMN_TILE_W * threadBlockColumns.y,
        dw * threadBlockColumns.y
    );

    cudaThreadSynchronize();

    // read back GPU result
    cudaMemcpy(output->elements, d_DataA, data_size, cudaMemcpyDeviceToHost);

}


float otherproduct(Matrix *input, Matrix *kernel, int x, int y) {
    float sum  = 0;


    for(int i = 0; i < kernel->height; i++) //Row
    {
        for(int j = 0; j < kernel->width; j++) //cols
        {
            if ((x + i < 0) || (x + i >= IMAGE_DIM) || (y + j < 0) || (y + j >= IMAGE_DIM)) {
                continue;
            }
            sum += input->elements[(x + i)*IMAGE_DIM + (y + j)]*kernel->elements[i*IMAGE_DIM + j];
        }

    }
    return sum/(IMAGE_DIM*IMAGE_DIM);
}


float calculate_overlap_product(Matrix *input, Matrix *kernel, int x, int y) {
    int x_min, y_min, x_max, y_max;
    int i, j;
    float result;

    if (x + kernel->width <= 0) {
        return 0;
    }
    if (x >= IMAGE_DIM) {
        return 0;
    }
    if (y >= kernel->height) {
        return 0;
    }
    if (y <= -IMAGE_DIM) {
        return 0;
    }

    if (x >= 0) {
        x_min = x;
    }
    else {
        x_min = 0;
    }

    if (x + kernel->width >= IMAGE_DIM) {
        x_max = IMAGE_DIM;
    }
    else {
        x_max = x + kernel->width;
    }

    if (y > 0) {
        y_max = 0;
    }
    else {
        y_max = y;
    }

    if (kernel->height - y > IMAGE_DIM) {
        y_min = -IMAGE_DIM;
    }
    else {
        y_min = y - kernel->height;
    }

    result = 0;

    for (i = x_min; i < x_max; i++) {

        for (j = y_min; j < y_max; j++) {

            result += input->elements[-j * IMAGE_DIM + i] * kernel->elements[(y - j) * kernel->width + i - x];
            if(result > 255*28*28)
            {
                printf("i %d j %d iEl %d kEl %d\n",i,j, -j * IMAGE_DIM + i, (y - j) * kernel->width + i - x);
            }
        }

    }

    return result/(IMAGE_DIM*IMAGE_DIM);

}


void convolve(Matrix *input, Matrix *kernel, Matrix *output) {

    int apron;

    int x, y;

    apron = kernel->width / 2;

    for (x = -apron; x < IMAGE_DIM - apron; x++) {
        for (y = -apron; y < IMAGE_DIM - apron; y++) {
            //printf("x %d y %d O %d\n",x, y ,(apron - y) * output->width + (x + apron));
            output->elements[(apron - y) * output->width + (x + apron)] = otherproduct(input, kernel, x, y);
            //calculate_overlap_product(input, kernel, x, y);

            if(output->elements[(apron - y) * output->width + (x + apron)] < 0)
            {
                printf("x %d y %d O %f \n",x,y, (apron - y) * output->width + (x + apron));
            }
        }

    }

}

void convolveTest(Matrix *input, Matrix *kernel, Matrix *output) {

    int apron;

    int x, y;

    apron = kernel->width / 2;

    for (x = -apron; x < IMAGE_DIM - apron; x++) {
        for (y = apron - IMAGE_DIM; y < apron; y++) {
            //printf("x %d y %d O %d\n",x, y ,(apron - y) * output->width + (x + apron));
            output->elements[(apron - y) * output->width + (x + apron)] = calculate_overlap_product(input, kernel, x, y);

            if(output->elements[(apron - y) * output->width + (x + apron)] < 0)
            {
                printf("x %d y %d\n",x,y);
            }
        }

    }

}
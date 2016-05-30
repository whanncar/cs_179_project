/* CPU NEURAL NETWORK, written by
 * Andrew and Wade, 2016
 */


#include "NeuralNetwork2.cuh"


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


void multiplyArrays(float * C, float * A, float * B, int r1, int c1, int r2, int c2)
{
	int ROW = r1;
	int COL = c2;
	int INNER = r2;
	int row = 0;
	int col = 0;
	int inner = 0;
	float sum = 0;

	if(c1 != r2)
	{
		printf("Error: Row - column mismatch\n");
	}
	printf("here\n");

	//int A[ROW][INNER], int B[INNER][COL], int C[ROW][COL];
	for (row = 0; row < ROW; ++row) 
	{
		/*if(row == 783 || row == 784)
			{
		printf("R %d C %d I %d S %f \n",row,col,inner,sum);
		}*/
 		for (col = 0; col < COL; ++col)
		{
			/*if(row == 783 || row == 784)
			{
				printf("R %d C %d I %d S %f \n",row,col,inner,sum);
			}*/
			sum = 0;
			for (inner = 0; inner < INNER; ++inner)
			{
				//sum += A[row][inner] * B[inner][col];
				sum += A[row*c1 + inner] * B[inner*c2 + col];
			}
			//C[row][col] = sum;
			C[row*COL + col] = sum;
			/*if(row == 783 || row == 784)
			{
				printf("R %d C %d I %d S %f \n",row,col,inner,sum);
			}*/
			
		}
		printf("R %d C %d I %d S %f \n",row,col,inner,sum);
	}
 }

//This works both for vectors and arrays as the same operation is 
//applied independantly to each element
void sigmodForward(float * output, float * input, int length)
{
	for(int i = 0; i < length; i++)
	{
		output[i] = 1/(1 + (float)exp( - input[i] ));
		//printf("%d\n",i);
	}
}

float randomFloat()
{
      float r = (float)rand()/(float)RAND_MAX;
      return r;
}

void initializeWeights(float * weights, int length)
{
	int k = 0;
	//loop through row, column and fill in with random initial weights
	for(k = 0; k < length; k ++);//row = num inputs + 1 bias
	{
		weights[k] = randomFloat();
		printf("k %d length %d\n",k,length);
	}
}

void findMax(float * labelEst, float *  Xout, int numTestSamples, int numOutputNeurons)
{
	float max;
	float maxIdx;
	int i, j;

	for(i = 0; i < numTestSamples; i++)
	{
		max = Xout[i];
		maxIdx = 0;
		//Find output that network estimates as most likely 
		//The last neuron is a residual bias unit that must be ignored
		for(j = 1; j < numOutputNeurons; j++)
		{
			if(max < Xout[numTestSamples * j  + i])
			{
				max = Xout[numTestSamples * j  + i];
				maxIdx = (float)j;
				
			}
			if(i < 10)
			{
				printf(" %f %f %f %d %d\n", max, maxIdx, Xout[numOutputNeurons * i  + j], i, j);
			}
		}
		/*if(i < 10)
		{
			printf("\n");
		}*/
		labelEst[i] = maxIdx;
	}
}

float calcLoss(float * labels, float * labelsEst, int numTestSamples)
{
	//Calculate loss
	float sum = 0;
	float loss = 0;

	for(int n = 0; n < numTestSamples; n++)
	{
		sum += (labels[n] - labelsEst[n])*(labels[n] - labelsEst[n]);
	}
	loss = (1/numTestSamples)*sum;
	return loss;
}


//dl size is transposed size of Xl
void calc_dl(float * dl, float * Xl, float * Y, int xl_r, int xl_c, int y_r, int y_c)
{
	float * C = (float *)malloc(xl_r*xl_c* sizeof(float));

	//trying to implement this
	//dl{l} = (2*(X{l} - ytr).*(X{l}).*(ones(size(X{l})) -X{l}))';
	oneMinusArray(C, Xl, xl_r, xl_c);
	subtractArrays(dl, Xl, Y, xl_r, xl_c);
	elementWiseMultiply(C, C, Xl, xl_r, xl_c);
	elementWiseMultiply(C, C, dl, xl_r, xl_c);
	scalarMultiply(C, C, 2.0, xl_r, xl_c);
	transpose(dl, C, xl_r,xl_c);

	free(C);
}

void elementWiseMultiply(float * C, float * A, float * B, int r, int c)
{
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			C[i*c + j] = A[i*c + j] * B[i*c + j]; 
		}
	}
}

void oneMinusArray(float * C, float * A, int r, int c)
{
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			C[i*c + j] = 1 - A[i*c + j];
		}
	}
}

void subtractArrays(float * C, float * A, float * B, int r, int c)
{
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			C[i*c + j] = A[i*c + j] - B[i*c + j];
		}
	}
}

void transpose(float * C, float * A, int r, int c)
{	
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			//transpose[d][c] = matrix[c][d];
			C[j*r + i] = A[i*c + j];
		}
	}
}

void calcDl_dw(data_matrix * dl_dw, data_matrix * dl, data_matrix * X, int currentLayer)
{
	float * dl_temp = (float *)malloc(dl[currentLayer].r * dl[currentLayer].c * sizeof(float));

	//Because X is of length NUM_layers +1, current layer is actually the l-1th layer
	float * Xlm1_temp = (float *)malloc(X[currentLayer].r * X[currentLayer].c * sizeof(float));

	printf("allocate temp arrays\n");
	transpose(dl_temp, dl[currentLayer].w, dl[currentLayer].r, dl[currentLayer].c);
	transpose(Xlm1_temp, X[currentLayer].w, X[currentLayer].r, X[currentLayer].c);
	printf("after Transpose\n");

	//dl_temp has size r = dl[l].c,  c = dl[l].r
	//xl_temp has size r = Xl[l].c  c = xl[l].r
	printf("r %d c %d r %d c %d\n",dl[currentLayer].c, dl[currentLayer].r, 
		X[currentLayer].c, X[currentLayer].r);
	multiplyArrays(dl_dw[currentLayer].w, dl_temp, Xlm1_temp, 
		dl[currentLayer].c, dl[currentLayer].r, 
		X[currentLayer].c, X[currentLayer].r);
	printf("afterMult\n");
	free(dl_temp);
	free(Xlm1_temp);
}

void scalarMultiply(float * C, float * A, float num, int r, int c)
{
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			//transpose[d][c] = matrix[c][d];
			C[j*r + i] = num*A[i*c + j];
		}
	}
}

void initialize_X(data_matrix * X, int * neuronsPerLayer, int numTestSamples, int numLayers)
{
	for(int i = 0; i < numLayers; i++)
	{
		if(i == 0)
		{
			X[i].r = neuronsPerLayer[i] + 1; //add bias term
			X[i].c = numTestSamples; 
			X[i].w = (float *)malloc(X[i].r * X[i].c * sizeof(float));
		}
		else
		{
			if(i == numLayers - 1) //if last layer don't have bias term
			{
				X[i].r = neuronsPerLayer[i- 1]; //No extra bias term
			}
			else
			{
				X[i].r = neuronsPerLayer[i-1] + 1; //add bias term
			}
			X[i].c = numTestSamples; 

			X[i].w = (float *)malloc(X[i].r * X[i].c* sizeof(float));
		}
	}
}

void initialize_W(data_matrix * weights, int * neuronsPerLayer, int numTestSamples)
{
	for(int i = 0; i < NUM_LAYERS + 1; i++)
	{
		weights[i].r = neuronsPerLayer[i]; 		//n_neurons

		if(i > 0)
		{
			weights[i].c = neuronsPerLayer[i-1] + 1; 	// n_neurons from previous layer + 1
		}
		else { //i = 0
			weights[i].c = neuronsPerLayer[i] + 1; 	// n_neurons from previous layer + 1
		}
		weights[i].w = (float *)malloc(weights[i].r * weights[i].c* sizeof(float));
	}
}

void initialize_layerSum(data_matrix *layerSum, int * neuronsPerLayer, int numTestSamples)
{
	for(int i = 0; i < NUM_LAYERS + 1; i++)
	{

		layerSum[i].r = neuronsPerLayer[i]; //no bias term
		layerSum[i].c = numTestSamples;

		layerSum[i].w = (float *)malloc(layerSum[i].r * layerSum[i].c* sizeof(float));
	}
}

void initialize_dl(data_matrix * dl, int * neuronsPerLayer, int numTestSamples)
{
	for(int i = 0; i < NUM_LAYERS + 1; i++)
	{
		dl[i].r = numTestSamples;
		dl[i].c = neuronsPerLayer[i];
		dl[i].w = (float *)malloc(dl[i].r * dl[i].c* sizeof(float));
	}
}

void initialize_dl_dw(data_matrix * dl_dw, int * neuronsPerLayer, int numTestSamples)
{
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		dl_dw[i].r = neuronsPerLayer[i]; 		//n_neurons

		if(i > 0)
		{
			dl_dw[i].c = neuronsPerLayer[i-1] + 1; 	// n_neurons from previous layer + 1
		}
		else { //i = 0
			dl_dw[i].c = neuronsPerLayer[i] + 1; 	// n_neurons from previous layer + 1
		}
		dl_dw[i].w = (float *)malloc(dl_dw[i].r * dl_dw[i].c* sizeof(float));
	}
}


void checkMem(void * pointer)
{
	if(pointer == NULL)
	{
		printf("Error memory pointer was NULL");
	}
}

void initialize_trainLabels(data_matrix * trainLabels, float * labels, int lastLayerNeurons, int numTestSamples)
{
	trainLabels->r = lastLayerNeurons;
	trainLabels->c = numTestSamples;
	trainLabels->w = (float *) malloc(trainLabels->r*trainLabels->c *sizeof(float));

	checkMem((void *)trainLabels->w);

	printf("beforeCreatelabels\n");
	for(int i = 0; i < trainLabels->c; i++)
	{
		//trainLabels[i*numClasses + labels[i]] = 1;
		//trainLabels[i*numCols + (int)labels[i]] = 1;
		//A little mind warping but this accesses by columns rather than rows
		//
		//printf("%d ",i);
		//printf("%d ",(int)labels[i]);
		//printf("%d ",i + (int)labels[i]*(trainLabels->c));
		if(i + (int)labels[i]*(trainLabels->c) < 0)
		{
			printf("lessthanzero\n");
		}
		trainLabels->w[i + (int)labels[i]*(trainLabels->c)] = 1;
		//printf("%d %f\n",i,trainLabels->w[i + (int)labels[i]*(trainLabels->c)]);
		//printf("%d %f\n",i,trainLabels[i + (int)labels[i]*cols]);
	}
	//createTrainLabels(trainLabels->w, labels, trainLabels->r, trainLabels->c);
}

//trainLabels must be allocated using calloc as this assume the array is already 0
void createTrainLabels(float *trainLabels, float * labels, int rows, int cols)
{
	
}

void free_data_matrix(data_matrix * X, int numLayers)
{
	for(int i = 0; i < numLayers; i++)
	{
		free(X[i].w);
	}
	free(X);
}

/*data_vector myVar;

typedef struct {

    int r;
    float * d;

} data_vector;

typedef struct {

    int r;
    int c;
    float * d;

} data_matrix;*/

/*
typedef struct {

    int num_rows;
    int num_cols;
    float ** d;

} data_matrix;

data_matrix * initializeArray(int r, int c)
{
	data_matrix *new_matrix;

	new_matrix = (data_matrix *) malloc(sizeof(data_matrix));

	new_matrix->num_rows = r;
	new_matrix->num_cols = c;

	new_matrix->d = (float **) malloc(r * sizeof(float *));

	for(int i = 0; i < r; i++)
	{
		new_matrix->d[i] = (float *) malloc(c * sizeof(float));
	}
}
*/

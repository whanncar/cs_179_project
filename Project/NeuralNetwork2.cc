//Andrew Janzen and Wade Hann-Caruthers
//5-17-2016
//Neural Network project


#include "NeuralNetwork2.cuh"



int main(int argc, char* argv[]) {

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
//char * fileName = "C:\Users\Andrew\GPU_files\Project\mnist_test.csv";

printf("numRows %d, numCols %d\n",numRows,numCols);

//System variables
char  buffer[2048] ;
char  *record, *line;
int   i = 0, j = 0, k = 0;
int   err  = -1;
int 	temp = 0;
int iter = 0;

//Network variables
//int numLayers = 3;
int *neuronsPerLayer = (int *) malloc(NUM_LAYERS *sizeof(int)); //[NUM_LAYERS];;
//Initialization not the most elegant 
neuronsPerLayer[0] = numCols;
neuronsPerLayer[1] = 90;
neuronsPerLayer[2] = 10;

float lambda = .005; //Learning rate
int numIterations = 100;
int numTestSamples = numRows;

data_matrix *weights = (data_matrix *) malloc(NUM_LAYERS *sizeof(data_matrix)); //[NUM_LAYERS];
data_matrix *X = (data_matrix *) malloc((NUM_LAYERS + 1) *sizeof(data_matrix)); ; //[NUM_LAYERS + 1];
data_matrix *layerSum = (data_matrix *) malloc(NUM_LAYERS *sizeof(data_matrix)); 
data_matrix *dl = (data_matrix *) malloc(NUM_LAYERS *sizeof(data_matrix)); ; 
data_matrix *trainLabels  = (data_matrix *) malloc(1 *sizeof(data_matrix)); ; 

printf("Initialize data and label Matrices\n");

float * data = (float *) malloc(numRows*(numCols + 1)* sizeof(float));
float * labels = (float *) malloc(numRows* sizeof(float));
float * labelEst = (float *) malloc(numRows* sizeof(float));
float * loss = (float *) malloc(numIterations* sizeof(float));



printf("\nInitialize Weight matrices\n");
initialize_W(weights, neuronsPerLayer, numTestSamples);

//Had trouble running this in a function - SEGV for some reason
for(int l = 0; l < NUM_LAYERS; l++)
{
	for(i = 0; i < weights[l].r * weights[l].c; i++)
	{
		weights[l].w[i] = randomFloat() -.5;
	}
}

printf("Read training data from File\n");
getDataFromFile(argv[1], labels,data,numRows, numCols);

printf("initalize X\n");
initialize_X(X, neuronsPerLayer, numTestSamples, NUM_LAYERS + 1);
X[0].w = data;
//free(data);

printf("initalize training labels\n");
initialize_trainLabels(trainLabels, labels, neuronsPerLayer[NUM_LAYERS - 1], numTestSamples);

printf("initalize dl\n");
initialize_dl(dl, neuronsPerLayer, numTestSamples);

printf("initalize layerSum\n");
initialize_layerSum(layerSum, neuronsPerLayer, numTestSamples);

//Labels must be initialize from getDataFromFile before calling trainLabels
for(int  l = 0; l  < NUM_LAYERS; l++)
{
	printf("Forward Neural network pass on layer %d\n",l);
	//layerSum[l].s = weights[l].w * data;
	//printf("w.r %d, w.c %d, X.r %d, X.c %d\n", weights[l].r, weights[l].c, X[l].r, X[l].c);
	//printf("layerSum.r %d, layersum.c %d\n",layerSum[l].r,layerSum[l].c);

	printf("\nMultiplying Arrays W and X at layer %d\n",l);
	multiplyArrays(layerSum[l].w, weights[l].w, X[l].w, weights[l].r, weights[l].c, X[l].r, X[l].c);

	//printf("completed layerSum[%d]",l);
	if((layerSum[l].r != X[l+1].r) && ( layerSum[l].c != X[l+1].c))
	{
		printf("Error:  dimension mismatch between layerSum[%d] and X[%d]\n",l,l+1);
		printf("layerSum.r %d, layerSum.c %c\n", layerSum[l].r, layerSum[l].c);
		printf("X[%d].r %d, X[%d].c %d\n", l+1, X[l+1].r, l+1, X[l+1].r);
	}
	
	
	if(X[l+1].r != layerSum[l].r && X[l+1].c != layerSum[l].c)
	{
		printf("error LayerSum dimensions do not match X at layer %d\n",l+1);

	}
	//printf("X.r %d X.c %d layerSum.r %d, layerSum.c %d\n",X[l+1].r, X[l+1].c, layerSum[l].r, layerSum[l].c);

	temp = layerSum[l].r*layerSum[l].c;
	//printf("numElements to sigmoid %d\n",temp);

	printf("Applying Sigmoid at layer %d\n",l);
	sigmodForward(X[l+1].w, layerSum[l].w, temp);
	
	printf("Adding 1's for bias\n");
	if(l + 1 < NUM_LAYERS)
	{
		for(int col = 0; col < X[l+1].c; col ++)
		{
			//fill in the last row of the output (next layers input) with the bias values
			X[l+1].w[ X[l+1].c * (X[l+1].r - 1) + col] = 1;
		}
	}
}

printf("First 10 samples of output X\n");
for(i = 0; i < X[NUM_LAYERS].r; i++)
{
	for(j = 0; j < 10; j++)
	{
		printf("%f  ",X[NUM_LAYERS].w[j + i*X[NUM_LAYERS].c]);
	}
	printf("\n");
}

//Find the most probable estimate of the output number based on the inputs
//This will be the neuron with the largest output value.  
//There are 4 x layers (input and 3 layers.)  Thus, X[numLayers] is the output layer
printf("numTestSamples %d, numoutputNeurons %d\n",numTestSamples,neuronsPerLayer[NUM_LAYERS - 1]);
findMax(labelEst, X[NUM_LAYERS].w, numTestSamples, neuronsPerLayer[NUM_LAYERS - 1]);

printf("Label estimate for first 10 samples\n");
for(i = 0; i < 10; i++)
{
	printf("%f  ", labelEst[i]);
}

printf("\nActual Label for first 10 samples\n");
for(i = 0; i < 10; i++)
{
	printf("%f  ",labels[i]);
}

//Calculate loss
loss[iter] = calcLoss(labels, labelEst, numTestSamples);
printf("loss %f\n",loss[iter]);

//Back propagate starting at the last layer
printf("Starting back prop\n");
//dl[l] = 2*(X[NUM_LAYERS].w - Y.w)*(1 - X[NUM_LAYERS])*X[NUM_LAYERS];

//Xl is column longer than it should be to subtract with Y
//due to the bias term I added

//Print layer rows and columns for debugging
for(i = 0; i < NUM_LAYERS; i++)
{
	printf("Layer %d\n", i);
	printf("dl r %d c %d\n",dl[i].r, dl[i].c);
	printf("X r %d c %d\n",X[i].r, X[i].c);
	printf("trainLabels r %d c %d\n",trainLabels->r, trainLabels->c);
}
printf("Layer %d\n", i);
printf("X r %d c %d\n",X[i].r, X[i].c);


calc_dl(dl[NUM_LAYERS - 1].w, X[NUM_LAYERS].w, trainLabels->w, X[NUM_LAYERS].r, 
	X[NUM_LAYERS].c, trainLabels->r, trainLabels->c);

printf("dl of layer %d\n", NUM_LAYERS - 1);

for(i = 0; i < 10; i++)
{
	for(j = 0; j < dl[NUM_LAYERS - 1].c; j++)
	{
		printf("%f  ", dl[NUM_LAYERS - 1].w[i*dl[NUM_LAYERS - 1].c + j]);
	}
	printf("\n");
}
/*
//dl_dw[l].w = dl* (ones X[l-1]);
void calcWeightChange(data_matrix *dl_dw, )


void oneMinusArray(float * C, float * A, int r, int c)
{
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			C[i*c + j] = 1 - A[i*c + j];
		}
	}
}*/



printf("free vectors\n");

free(data);
printf("free labels\n");
free(labels);
printf("free labelEst\n");
free(labelEst);
//printf("free loss\n");
//free(loss);
printf("Free struct arrays\n");

/*
for(i = 0; i < NUM_LAYERS; i++)
	{
		free(weights[i].w);
	}
free(weights);

printf("after");
free_data_matrix(weights, NUM_LAYERS );
printf("here");
free_data_matrix(layerSum, NUM_LAYERS );
printf("here");
free_data_matrix(dl, NUM_LAYERS );
printf("here");
free_data_matrix(trainLabels, 1);
printf("here");
free_data_matrix(X, NUM_LAYERS + 1);
*/
}
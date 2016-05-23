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
int temp = 0;

//Network variables
//int numLayers = 3;
int neuronsPerLayer[NUM_LAYERS] = {numCols, 90, 10};
float lambda = .005; //Learning rate
int numIterations = 100;
int numTestSamples = numRows;
struct Array *weights = (Array *) malloc(NUM_LAYERS *sizeof(Array)); //[NUM_LAYERS];
struct Array *X = (Array *) malloc((NUM_LAYERS + 1) *sizeof(Array)); ; //[NUM_LAYERS + 1];
struct Array *layerSum = (Array *) malloc(NUM_LAYERS *sizeof(Array)); 
struct Array *dl = (Array *) malloc(NUM_LAYERS *sizeof(Array)); ; 
struct Array trainLabels;

printf("Initialize data and label Matrices\n");
float * data = (float *) malloc(numRows*(numCols + 1)* sizeof(float));
float * labels = (float *) malloc(numRows* sizeof(float));
float * labelEst = (float *) malloc(numRows* sizeof(float));
float * loss = (float *) malloc(numIterations* sizeof(float));

int iter = 0;

//Configure label array which is same size as output data  - used for back prop
trainLabels.r = neuronsPerLayer[NUM_LAYERS - 1];
trainLabels.c = numTestSamples;
trainLabels.w = (float *) malloc(trainLabels.r*trainLabels.c *sizeof(float));


createTrainLabels(trainLabels.w, labels, trainLabels.r, trainLabels.c);

initialize_X(X, neuronsPerLayer, numTestSamples);
initialize_W(weights, neuronsPerLayer, numTestSamples);
initialize_dl(dl, neuronsPerLayer, numTestSamples);
initialize_layerSum(layerSum, neuronsPerLayer, numTestSamples);
/*
//Create Xin struct for convinience 
for(i = 0; i < NUM_LAYERS + 1; i++)
{
	if(i == 0)
	{
		X[i].r = neuronsPerLayer[i] + 1; //add bias term
		X[i].c = numTestSamples; 
		X[i].w = (float *)malloc(X[i].r * X[i].c * sizeof(float));
	}

	if(i == NUM_LAYERS - 1) //if last layer don't have bias term
	{
		X[i+1].r = neuronsPerLayer[i]; //No extra bias term
	}
	else
	{
		X[i+1].r = neuronsPerLayer[i] + 1; //add bias term
	}
	X[i+1].c = numTestSamples; 
	X[i+1].w = (float *)malloc(X[i+1].r * X[i+1].c* sizeof(float));

	layerSum[i].r = neuronsPerLayer[i]; //no bias term
	layerSum[i].c = numTestSamples;

	weights[i].r = neuronsPerLayer[i]; 		//n_neurons
	dl[i].r = numTestSamples;
	dl[i].c = neuronsPerLayer[i];

	if(i > 0)
	{
		weights[i].c = neuronsPerLayer[i-1] + 1; 	// n_neurons from previous layer + 1
	}
	else { //i = 0
		weights[i].c = neuronsPerLayer[i] + 1; 	// n_neurons from previous layer + 1
	}

	dl[i].w = (float *)malloc(dl[i].r * dl[i].c* sizeof(float));
	weights[i].w = (float *)malloc(weights[i].r * weights[i].c* sizeof(float));
	layerSum[i].w = (float *)malloc(layerSum[i].r * layerSum[i].c* sizeof(float));

	printf("layer %d, w.r %d, w.c %d\n",i,weights[i].r,weights[i].c);
	printf("layer %d, X.r %d, X.c %d\n",i, X[i].r,X[i].c);
	printf("layer %d, LS.r %d, LS.c %d\n",i,layerSum[i].r,layerSum[i].c);
	if(i == 2)
	{
		i = 3;
		printf("layer %d, X.r %d, X.c %d\n",i, X[i].r,X[i].c);
	}
}
*/

X[0].w = data;

printf("\nInitialize Weight matrices\n");
//Initialize the first weight layer

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
//printf("got data from file\n");



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
	for(int col = 0; col < X[l+1].c; col ++)
	{
		//fill in the last row of the output (next layers input) with the bias values
		X[l+1].w[ X[l+1].c * (X[l+1].r - 1) + col] = 1;
	}
}

//Find the most probable estimate of the output number based on the inputs
//This will be the neuron with the largest output value.  
//There are 4 x layers (input and 3 layers.)  Thus, X[numLayers] is the output layer
printf("numTestSamples %d, numoutputNeurons %d\n",numTestSamples,neuronsPerLayer[NUM_LAYERS - 1]);
findMax(labelEst, X[NUM_LAYERS].w, numTestSamples, neuronsPerLayer[NUM_LAYERS - 1]);

//Calculate loss
loss[iter] = calcLoss(labels, labelEst, numTestSamples);
printf("loss %f\n",loss[iter]);

//Back propagate starting at the last layer
printf("Starting back prop\n");
//dl[l] = 2*(X[NUM_LAYERS].w - Y.w)*(1 - X[NUM_LAYERS])*X[NUM_LAYERS];

//Xl is column longer than it should be to subtract with Y
//due to the bias term I added

calc_dl(dl[NUM_LAYERS - 1].w, X[NUM_LAYERS].w, trainLabels.w, X[NUM_LAYERS].r, 
	X[NUM_LAYERS].c, trainLabels.r, trainLabels.c);

printf("dl of layer %d\n", NUM_LAYERS - 1);

for(i = 0; i < 100; i++)
{
	printf("%f  ", dl[NUM_LAYERS - 1].w[i]);
	if(i % 5 == 0)
	{
		printf("\n");
	}
}



free(data);
free(labels);

}
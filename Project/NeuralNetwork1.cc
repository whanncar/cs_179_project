//Andrew Janzen and Wade Hann-Caruthers
//5-17-2016
//Neural Network project

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curand.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <cstdio>
#include <cuda_runtime.h>
#include <time.h>
#include <curand.h>
#include <cassert>


#define IMG_LEN (784)
#define NUM_LAYERS 3


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

struct Weight{
   int  m_inputs;
   int  n_neurons;
   float * w;
};


//Actual length of data is (numCols + 1) * (numRows)
void getDataFromFile(char * inputFile, float * labels, float * data, int numCols, int numRows)
{
	char  buffer[2048] ;
	char  *record, *line;
	int   i = 0, j = 0;
	int   err  = -1;
	FILE *fstream  = fopen(inputFile,"r");

	printf("Read training data from File\n");
	while(((line=fgets(buffer,sizeof(buffer),fstream))!=NULL) && (i < numRows))
	{
		record = strtok(line,",");
		j = 0;
		//Read label for each image //label is the first data point in a row
		labels[i] = atoi(record);
		record = strtok(NULL,",");
		printf("%d %d  ",i,labels[i]);

		while(record != NULL && (j < numCols))  
		{
			data[i*(numCols + 1) + j] = atoi(record);
			record = strtok(NULL,",");
			j++;
			if(j == numCols)
			{
				data[i*(numCols + 1) + j] = 1;  //Add one extra element at the end for a bias
				printf("  %f", data[i*(numCols + 1) + j]);
			}
		}
		printf("%d %d\n", j-1, data[i*(numCols + 1) + j-1] );
		++i ;
	}

	err = fclose(fstream);
	if(err != 0)
	{
		printf("Error closing file");
	}
}



void createTrainLabels(float *trainLabels, float * labels, int numRows, int numClasses)
{
	for(int i = 0; i < numRows; i++)
	{
			trainlabels[i*numClasses + labels[i]] = 1;
	}
}
void sigmod(*)


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

//System variables
char  buffer[2048] ;
char  *record, *line;
int   i = 0, j = 0;
int   err  = -1;

//Network variables
//int numLayers = 3;
int neuronsPerLayer[NUM_LAYERS] = {numCols, 90, 10};
float lambda = .005; //Learning rate
int numIterations = 100;
int numTestSamples = numRows;
struct Weight weights[NUM_LAYERS];


//Initialize the first weight layer
weights[0].m_inputs  = neuronsPerLayer[0] + 1;  //The added 1 is for the constant bias term
weights[0].n_neurons = neuronsPerLayer[0]; //Any number can be chosen here
weights[0].w 		 = (float *)calloc(weights[0].m_inputs * weights[0].n_neurons, sizeof(float));

int k = 0;
int n = 0;
int m = 0;

printf("Initialize Weight matrices\n");

//Allocate memory and zero
float * trainLabels = (float *) calloc(numClasses*numRows* sizeof(float));

/*
//initialize the kth weight layer
for(k = 1; k < NUM_LAYERS; k++)
{
	printf("beginning");
	weights[k].m_inputs  = weights[k-1].n_neurons + 1;  //Comes from first layer - cant change this
	weights[k].n_neurons = neuronsPerLayer[k]; //Any number can be chosen here
	printf("k: %d, m_inputs: %d, n_neurons_ %d",k, weights[k].m_inputs, weights[k].n_neurons);
	weights[k].w 		 = (float *)calloc(weights[k].m_inputs * weights[k].n_neurons, sizeof(float));
	if (weights[k].w == 0)
	{
		printf("ERROR: Out of memory\n");
		return 1;
	}

	printf("here");
	//loop through row, column and fill in with random initial weights
	for(m = 0; m < weights[k].m_inputs; m ++);//row = num inputs + 1 bias
	{
		for(n = 0; n < weights[k].n_neurons; n ++) //column = num Neurons
		{
			weights[k].w[m*weights[k].n_neurons + n] = (float)rand()/(float)(RAND_MAX);
		}
	}
	printf("there");
	printf("weights[%d] %d %d %f\n",k,weights[k].m_inputs, 
		weights[k].n_neurons, weights[k].w[1]);
}
*/

float * data = (float *) malloc((numCols + 1)*numRows* sizeof(float));
if (data == 0)
{
	printf("ERROR: Out of memory\n");
	return 1;
}

float * labels = (float *) malloc(numRows* sizeof(float));
if (labels == 0)
{
	printf("ERROR: Out of memory\n");
	return 1;
}
/*

FILE *fstream  = fopen(argv[1],"r");

if(fstream == NULL)
{
	printf("\n file opening failed ");
	return -1 ;
}*/


printf("Read training data from File\n");
getDataFromFile(argv[1], labels,data,numCols,numRows);

/*
while(((line=fgets(buffer,sizeof(buffer),fstream))!=NULL) && (i < numRows))
{
		record = strtok(line,",");
		j = 0;
		//Read label for each image //label is the first data point in a row
		labels[i] = atoi(record);
		record = strtok(NULL,",");
		printf("%d %d  ",i,labels[i]);

		while(record != NULL && (j < numCols))  
		{
			data[i*numCols + j] = atoi(record);
			record = strtok(NULL,",");
			j++;
		}
		printf("%d %d\n", j-1, data[i*numCols + j-1] );
		++i ;
}*/


free(data);
free(labels);
/*
err = fclose(fstream);
if(err != 0)
{
	printf("Error closing file");
}
*/



}
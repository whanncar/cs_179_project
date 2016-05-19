//Andrew Janzen and Wade Hann-Caruthers
//5-17-2016
//Neural Network project

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define IMG_LEN (784)


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

int main(int argc, char* argv[]) {
  

  if (argc < 5){
      printf("Usage: (threads per block) (max number of blocks) \n
      	(numInputElements) (trainingFilename) \n");
      exit(-1);
  }

  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);

  const char trainingFilename = atoi(argv[2]);

    FILE* stream = fopen(trainingFilename, "r");

    char line[1024];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        printf("Field 3 would be %s\n", getfield(tmp, 3));
        // NOTE strtok clobbers tmp
        free(tmp);
    }
}
#include <stdio.h>
#include <stdlib.h>
#include "neural_net/neural_net.cuh"


neural_net *nn;
neural_net *nn_dev;
sample_set *samples;
sample_set *samples_dev;

char usage[] = "usage: <number of samples> <sample length> <label length> "
              "<number of hidden layers> <layer 1 size> ... <layer n size>\n";

void initialize_neural_net(int, char **);
void initialize_samples(int, char **);
void print_output(neural_net *);
void print_weights(neural_net *);

int main(int argc, char **argv) {

    if (argc == 1) {
        printf(usage);
        return 0;
    }

    initialize_neural_net(argc, argv);

    return 0;
}



void initialize_neural_net(int argc, char **argv) {

    int num_layers;
    int num_inputs;
    int input_size;
    int output_size;
    int *layer_weight_specs;
    int i;   

    num_inputs = atoi(argv[1]);
    input_size = atoi(argv[2]);
    output_size = atoi(argv[3]);
    num_layers = atoi(argv[4]);

    layer_weight_specs = (int *) malloc(num_layers * sizeof(int));

    for (i = 0; i < num_layers; i++) {
        layer_weight_specs[i] = atoi(argv[5 + i]);
    }

    num_layers += 1;

    nn = gpu_new_neural_net(num_layers, num_inputs, input_size, output_size, layer_weight_specs);

    free(layer_weight_specs);
}



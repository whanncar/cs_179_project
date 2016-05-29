#include <stdio.h>
#include <stdlib.h>
#include "./neural_net/neural_net.h" 
#include "io_utils/fileio.h"


neural_net *nn;
sample_set *samples;

char usage[] = "usage: <number of samples> <sample length> <label length> "
              "<number of hidden layers> <layer 1 size> ... <layer n size>\n";

void initialize_neural_net(int, char **);
void initialize_samples(int, char **);

int main(int argc, char **argv) {

    float loss;
    float lambda;

    lambda = .1;

    if (argc == 1) {
        printf(usage);
        return 0;
    }


    initialize_neural_net(argc, argv);
    initialize_samples(argc, argv);


    while (1) {

        train_neural_net(nn, samples, lambda);
        loss = calculate_loss(nn, samples); 
        printf("%f, %f\n", loss, calculate_percent_predicted_correctly(nn, samples));
    }

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

    nn = new_neural_net(num_layers, num_inputs, input_size, output_size, layer_weight_specs);

    free(layer_weight_specs);
}


void initialize_samples(int argc, char **argv) {

    samples = get_samples_from_file("training_data/mnist_test.csv",
                                    atoi(argv[1]), atoi(argv[2]));
}

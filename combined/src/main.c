#include "main.h"

#define NUM_ITERATIONS 1000
#define LAMBDA .0001

neural_net *nn;
neural_net *nn_dev;
sample_set *samples;
sample_set *samples_dev;

char usage[] = "usage: <number of samples> <sample length> <label length> "
              "<number of hidden layers> <layer 1 size> ... <layer n size>\n";


int main(int argc, char **argv) {

    int iteration;

    if (argc == 1) {
        printf(usage);
        return 0;
    }

    initialize_neural_net(argc, argv);
    gpu_initialize_neural_net(argc, argv);
    copy_neural_net_to_gpu(nn, nn_dev);

    initialize_samples(argc, argv);
    gpu_initialize_samples();

    gpu_set_neural_net_input(nn_dev, samples_dev);

    for (iteration = 0; iteration < NUM_ITERATIONS; iteration++) { 

        gpu_train_neural_net(nn_dev, samples_dev, LAMBDA);

        printf("Iteration: %d\nLoss: %d\n\n", iteration,
               gpu_calculate_loss(nn_dev, samples_dev));
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

    initialize_neural_net_weights(nn);

    free(layer_weight_specs);
}


void gpu_initialize_neural_net(int argc, char **argv) {

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

    nn_dev = gpu_new_neural_net(num_layers, num_inputs,
                                input_size, output_size,
                                layer_weight_specs);

    free(layer_weight_specs);
}


void initialize_samples(int argc, char **argv) {

    samples = get_samples_from_file("training_data/mnist_test.csv",
                                    atoi(argv[1]), atoi(argv[2]));
}


void gpu_initialize_samples() {

    samples_dev = (sample_set *) malloc(sizeof(sample_set));

    samples_dev->sample_labels = 
        gpu_new_matrix(samples->sample_labels->num_rows,
                       samples->sample_labels->num_cols);

    samples_dev->sample_inputs = 
        gpu_new_matrix(samples->sample_inputs->num_rows,
                       samples->sample_inputs->num_cols);

    cudaMemcpy(samples_dev->sample_labels->data,
               samples->sample_labels->data,
               samples->sample_labels->num_rows *
               samples->sample_labels->num_cols *
               sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(samples_dev->sample_inputs->data,
               samples->sample_inputs->data,
               samples->sample_inputs->num_rows *
               samples->sample_inputs->num_cols *
               sizeof(float),
               cudaMemcpyHostToDevice);
}

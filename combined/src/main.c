#include "main.h"

neural_net *nn;
neural_net *nn_dev;
sample_set *samples;
sample_set *samples_dev;

char usage[] = "usage: <number of samples> <sample length> <label length> "
              "<number of hidden layers> <layer 1 size> ... <layer n size>\n";

void initialize_neural_net(int, char **);
void gpu_initialize_neural_net(int, char **);
void initialize_samples(int, char **);
void gpu_initialize_samples();
void print_output(neural_net *);
void print_weights(neural_net *);

int main(int argc, char **argv) {

    float loss;
    float lambda;

    lambda = .0005;

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

    train_neural_net(nn, samples, lambda);
    gpu_train_neural_net(nn_dev, lambda);

    printf("%f\n", calculate_loss(nn, samples));
    printf("%f\n", gpu_calculate_loss(nn_dev, samples_dev));

/*
    while (1) {

        train_neural_net(nn, samples, lambda);
        loss = calculate_loss(nn, samples);

        print_output(nn);
 
        print_weights(nn);

        printf("%f, %f\n", loss, calculate_percent_predicted_correctly(nn, samples));
    }
*/
    return 0;
}

void print_weights(neural_net *net) {

    int i, j, k;
    neural_layer *layer;


    for (k = 0; k < net->num_layers; k++) {

        layer = net->layer_ptrs[k];

        for (i = 0; i < layer->w->num_rows; i++) {

            for (j = 0; j < layer->w->num_cols; j++) {

                printf("%f ", layer->w->data[i * layer->w->num_cols + j]);

            }

            printf("\n");

        }

        printf("\n");

    }

}



void print_output(neural_net *net) {

    data_matrix *output;

    output = net->output;

    int i, j;

    int rows, cols;

    rows = output->num_rows;
    cols = output->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < 10; j++) {

            printf("%f ", output->data[i * cols + j]);

        }

        printf("\n");

    }

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

    nn_dev = gpu_new_neural_net(num_layers, num_inputs, input_size, output_size, layer_weight_specs);

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

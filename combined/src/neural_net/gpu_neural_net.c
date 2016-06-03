#include <stdio.h>
#include "gpu_neural_net.h"
#include <cuda.h>
#include <cuda_runtime.h>


neural_net *gpu_new_neural_net(int num_layers, int num_inputs,
                               int input_size, int output_size,
                               int *layer_weight_specs) {

    int i;

    neural_net *nn;

    nn = (neural_net *) malloc(sizeof(neural_net));

    nn->num_layers = num_layers;

    nn->layer_ptrs =
        (neural_layer **) malloc(num_layers * sizeof(neural_layer *));


printf("a\n");

    /* Make layers */
    nn->layer_ptrs[0] = gpu_new_neural_layer(input_size,
                                             layer_weight_specs[0],
                                             num_inputs);

printf("b\n");

    for (i = 1; i < num_layers - 1; i++) {
        nn->layer_ptrs[i] = gpu_new_neural_layer(layer_weight_specs[i - 1],
                                                 layer_weight_specs[i],
                                                 num_inputs);

printf("c\n");

    }


    nn->layer_ptrs[i] = gpu_new_neural_layer(layer_weight_specs[i - 1],
                                             output_size,
                                             num_inputs);


printf("d\n");

    /* Connect inputs and outputs of adjacent layers */
    nn->layer_ptrs[0]->input = gpu_new_matrix(input_size, num_inputs);

    for (i = 1; i < num_layers; i++) {
        nn->layer_ptrs[i]->input = nn->layer_ptrs[i - 1]->output;
    }


    nn->input = nn->layer_ptrs[0]->input;
    nn->output = nn->layer_ptrs[num_layers - 1]->output;

    return nn;

}



void gpu_free_neural_net(neural_net *nn) {

    int i;

    cudaFree(nn->input);

    for (i = 0; i < nn->num_layers; i++) {
        gpu_free_neural_layer(nn->layer_ptrs[i]);
    }

    free(nn->layer_ptrs);

    free(nn);

}




void copy_neural_net_to_gpu(neural_net *nn, neural_net *nn_dev) {

    int i;

    cudaMemcpy(nn_dev->input->data, nn->input->data,
               nn->input->num_rows *
               nn->input->num_cols *
               sizeof(float), cudaMemcpyHostToDevice);

    for (i = 0; i < nn->num_layers; i++) {
        copy_neural_layer_to_gpu(nn->layer_ptrs[i], nn_dev->layer_ptrs[i]);
    }

}




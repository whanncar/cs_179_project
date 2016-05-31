#include "neural_net.h"


neural_net *new_neural_net(int num_layers, int num_inputs,
                           int input_size, int output_size,
                           int *layer_weight_specs) {

    int i;

    neural_net *nn;

    nn = (neural_net *) malloc(sizeof(neural_net));

    nn->num_layers = num_layers;

    nn->layer_ptrs =
        (neural_layer **) malloc(num_layers * sizeof(neural_layer *));


    /* Make layers */
    nn->layer_ptrs[0] = new_neural_layer(input_size,
                                         layer_weight_specs[0],
                                         num_inputs);

    for (i = 1; i < num_layers - 1; i++) {
        nn->layer_ptrs[i] = new_neural_layer(layer_weight_specs[i - 1],
                                             layer_weight_specs[i],
                                             num_inputs);
    }

    nn->layer_ptrs[i] = new_neural_layer(layer_weight_specs[i - 1],
                                         output_size,
                                         num_inputs);


    /* Connect inputs and outputs of adjacent layers */
    nn->layer_ptrs[0]->input = new_matrix(input_size, num_inputs);

    for (i = 1; i < num_layers; i++) {
        nn->layer_ptrs[i]->input = nn->layer_ptrs[i - 1]->output;
    }


    nn->input = nn->layer_ptrs[0]->input;
    nn->output = nn->layer_ptrs[num_layers - 1]->output;

    return nn;

}


void initialize_neural_net_weights(neural_net *nn) {

    int i;
    neural_layer *layer;

    for (i = 0; i < nn->num_layers; i++) {

        layer = nn->layer_ptrs[i];

        fill_matrix_rand(layer->w, -.5, .5);

        compute_matrix_transpose(layer->w, layer->w_T);

    }

}

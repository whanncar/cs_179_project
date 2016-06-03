#include "neural_layer.h"


neural_layer *new_neural_layer(int input_length,
                               int num_weights,
                               int num_inputs) {

    neural_layer *layer;

    layer = (neural_layer *) malloc(sizeof(neural_layer));

    layer->w = new_matrix(num_weights, input_length);

    layer->w_T = new_matrix(input_length, num_weights);

    layer->s = new_matrix(num_weights, num_inputs);

    layer->output = new_matrix(num_weights, num_inputs);

    layer->dL_ds = new_matrix(num_weights, num_inputs);

    layer->dL_dw = new_matrix(num_weights, input_length);

    return layer;

}

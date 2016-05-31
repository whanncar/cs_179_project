#include "neural_layer.h"


neural_layer *gpu_new_neural_layer(int input_length,
                                   int num_weights,
                                   int num_inputs) {

    neural_layer *layer;

    layer = (neural_layer *) malloc(sizeof(neural_layer));

    layer->w = gpu_new_matrix(num_weights, input_length);

    layer->w_T = gpu_new_matrix(input_length, num_weights);

    layer->s = gpu_new_matrix(num_weights, num_inputs);

    layer->output = gpu_new_matrix(num_weights, num_inputs);

    layer->dL_ds = gpu_new_matrix(num_weights, num_inputs);

    layer->dL_dw = gpu_new_matrix(num_weights, input_length);

    return layer;

}

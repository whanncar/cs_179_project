#include "../utils/utils.h"
#include "neural_net.h"
#include <math.h>



/*
 * sigmoid_filter: Computes the sigmoid function
 *
 * arguments: x: Input value to sigmoid
 *
 * return value: Output value from sigmoid
 *
 * where should I live? UNRESOLVED
 *
 */

float sigmoid_filter(float x) {

    return 1 / (1 + expf(-x));

}



/*
 * forward_propagate_layer: Forward propagates the given neural layer
 *                          using the given filter
 *
 * arguments: layer: The neural layer to forward propagate
 *
 */

void forward_propagate_layer(neural_layer *layer) {

    /* Calculate pre-raw weighted sums */
    calculate_matrix_times_vector(layer->w, layer->input, layer->r);

    /* Calculate raw weighted sums */
    multiply_vectors_componentwise(layer->r, layer->t, layer->s);

    /* Calculate output by filtering raw weighted sums */
    apply_filter_to_vector_componentwise(layer->s,
                                         &sigmoid_filter,
                                         layer->output);

}



/*
 * forward_propagate_neural_net: Forward propagates the given neural net
 *                               using the given filter
 *
 * arguments: nn: The neural net to be forward propagated
 *
 */

void forward_propagate_neural_net(neural_net *nn) {

    int i;

    /* Propagate each layer */
    for (i = 0; i < nn->num_layers; i++) {
        forward_propagate_layer(nn->layer_ptrs[i]);
    }

}



/*
 * UNRESOLVED
 *
 */

void calculate_dL_ds_layer(neural_layer *layer,
                           neural_layer *next_layer) {

    multiply_vectors_componentwise(next_layer->dL_ds,
                                   next_layer->t,
                                   layer->dL_ds);

    compute_matrix_times_vector(next_layer->w_T,
                                layer->dL_ds,
                                layer->dL_ds);

    multiply_vectors_componentwise(next_layer->input,
                                   layer->dL_ds,
                                   layer->dL_ds);

    compute_additive_inverse_of_vector(next_layer->input,
                                       next_layer->input);

    add_constant_componentwise_to_vector(next_layer->input, 1,
                                         next_layer->input);

    multiply_vectors_componentwise(next_layer->input,
                                   layer->dL_ds,
                                   layer->dL_ds);

    add_constant_componentwise_to_vector(next_layer->input, -1,
                                         next_layer->input);

    compute_additive_inverse_of_vector(next_layer->input,
                                       next_layer->input);

}



/*
 * UNRESOLVED
 *
 */

void compute_dL_ds_last_layer(neural_net *nn, data_vector *expected_output) {
 
    neural_layer *last_layer;

    last_layer = nn->layer_ptrs[nn->num_layers - 1];


    compute_additive_inverse_of_vector(expected_output, expected_output);

    add_vectors(last_layer->output, expected_output, last_layer->dL_ds);

    multiply_vector_by_constant(last_layer->dL_ds, 2, last_layer->dL_ds);

    multiply_vectors_componentwise(last_layer->dL_ds,
                                   last_layer->output,
                                   last_layer->dL_ds);

    compute_additive_inverse_of_vector(last_layer->output,
                                       last_layer->output);

    add_constant_componentwise_to_vector(last_layer->output, 1,
                                         last_layer->output);

    multiply_vectors_componentwise(last_layer->output,
                                   last_layer->dL_ds,
                                   last_layer->dL_ds);

    add_constant_componentwise_to_vector(last_layer->output, -1,
                                         last_layer->output);

    compute_additive_inverse_of_vector(last_layer->output,
                                       last_layer->output);
}

/*
 * UNRESOLVED
 *
 */

void compute_dL_ds_all_layers(neural_net *nn,
                              data_vector *expected_output) {

    int i;

    compute_dL_ds_last_layer(nn, expected_output);

    for (i = nn->num_layers - 2; i >= 0; i--) {
        compute_dL_ds_layer(nn->layer_ptrs[i], nn->layer_ptrs[i + 1]);
    }
}


/*
 * UNRESOLVED
 *
 */

void update_t_layer(neural_layer *layer, float step) {

    multiply_vectors_componentwise(layer->dL_ds, layer->r, layer->dL_ds);

    multiply_vector_by_constant(layer->dL_ds, step, layer->dL_ds);

    compute_additive_inverse_of_vector(layer->dL_ds, layer->dL_ds);

    add_vectors(layer->t, layer->dL_ds, layer->t);

}



/*
 * UNRESOLVED
 *
 */

void update_t_all_layers(neural_net *nn, float step) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {
        update_t_layer(nn->layer_ptrs[i], step);
    }

}



/*
 * UNRESOLVED
 *
 */

void backward_propagate_neural_net(neural_net *nn,
                                   data_vector *expected_output,
                                   float step) {

    compute_dL_ds_all_layers(nn, expected_output);

    update_t_all_layers(nn, step);

}

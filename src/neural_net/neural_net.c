#include <stdio.h>
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
    calculate_matrix_times_vector(layer->w, layer->input, layer->s);

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

    calculate_matrix_times_vector(next_layer->w_T,
                                next_layer->dL_ds_local,
                                layer->dL_ds_local);

    multiply_vectors_componentwise(next_layer->input,
                                   layer->dL_ds_local,
                                   layer->dL_ds_local);

    compute_additive_inverse_of_vector(next_layer->input,
                                       next_layer->input);

    add_constant_componentwise_to_vector(next_layer->input, 1,
                                         next_layer->input);

    multiply_vectors_componentwise(next_layer->input,
                                   layer->dL_ds_local,
                                   layer->dL_ds_local);

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

    add_vectors(last_layer->output, expected_output, last_layer->dL_ds_local);

    compute_additive_inverse_of_vector(expected_output, expected_output);

    multiply_vector_by_constant(last_layer->dL_ds_local, 2, last_layer->dL_ds_local);

    multiply_vectors_componentwise(last_layer->dL_ds_local,
                                   last_layer->output,
                                   last_layer->dL_ds_local);

    compute_additive_inverse_of_vector(last_layer->output,
                                       last_layer->output);

    add_constant_componentwise_to_vector(last_layer->output, 1,
                                         last_layer->output);

    multiply_vectors_componentwise(last_layer->output,
                                   last_layer->dL_ds_local,
                                   last_layer->dL_ds_local);

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
        calculate_dL_ds_layer(nn->layer_ptrs[i], nn->layer_ptrs[i + 1]);
    }
}



/*
 * UNRESOLVED
 *
 */

void add_dL_ds_local_to_dL_ds_global_all_layers(neural_net *nn) {

    int i;
    neural_layer *current_layer;

    for (i = 0; i < nn->num_layers; i++) {

        current_layer = nn->layer_ptrs[i];

        add_vectors(current_layer->dL_ds_local,
                    current_layer->dL_ds_global,
                    current_layer->dL_ds_global);

    }

}



/*
 * UNRESOLVED
 *
 */

void update_t_layer(neural_layer *layer, float step) {

    multiply_vectors_componentwise(layer->dL_ds_global,
                                   layer->r, layer->dL_ds_global);

    multiply_vector_by_constant(layer->dL_ds_global,
                                step,
                                layer->dL_ds_global);

    compute_additive_inverse_of_vector(layer->dL_ds_global,
                                       layer->dL_ds_global);

    add_vectors(layer->t, layer->dL_ds_global, layer->t);

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

void backward_propagate_neural_net_single_sample(neural_net *nn,
                                                 data_vector *expected_output) {

    compute_dL_ds_all_layers(nn, expected_output);

    add_dL_ds_local_to_dL_ds_global_all_layers(nn);

}



/*
 * UNRESOLVED
 *
 */

void set_neural_net_input(neural_net *nn, data_vector *input) {

    int i;

    for (i = 0; i < nn->input->size; i++) {
        nn->input->data[i] = input->data[i];
    }

}


/*
 * UNRESOLVED
 *
 */

void train_neural_net(neural_net *nn, sample_set *set, float step) {

    int i;
    sample *current_sample;

    for (i = 0; i < set->num_samples; i++) {

        current_sample = set->sample_ptrs[i];

        set_neural_net_input(nn, current_sample->input);

        forward_propagate_neural_net(nn);

        backward_propagate_neural_net_single_sample(nn,
            current_sample->expected_output);
    }

    update_t_all_layers(nn, step);

}



/*
 * UNRESOLVED
 *
 */

float calculate_sample_loss(neural_net *nn, sample *s) {

    int i;
    float result;
    float diff;

    result = 0;

    set_neural_net_input(nn, s->input);

    forward_propagate_neural_net(nn);

    for (i = 0; i < nn->output->size; i++) {

        diff = nn->output->data[i] - s->expected_output->data[i];

        result += diff * diff;

    }

    return result;

}



/*
 * UNRESOLVED
 *
 */

int predict(neural_net *nn, sample *s) {

    int i, max_index;
    float max;

    set_neural_net_input(nn, s->input);

    forward_propagate_neural_net(nn);

    max = nn->output->data[0];
    max_index = 0;

    for (i = 1; i < nn->output->size; i++) {

        if (nn->output->data[i] > max) {

            max = nn->output->data[i];
            max_index = i;

        }

    }

    return max_index;

}



/*
 * UNRESOLVED
 *
 */

float calculate_loss(neural_net *nn, sample_set *set) {

    int i;
    float result;

    result = 0;

    for (i = 0; i < set->num_samples; i++) {

        result += calculate_sample_loss(nn, set->sample_ptrs[i]);

    }

    return result;

}



/*
 * UNRESOLVED
 *
 */

float calculate_percent_predicted_correctly(neural_net *nn, sample_set *set) {

    int i;
    int prediction;
    float count;

    count = 0;

    for (i = 0; i < set->num_samples; i++) {

        prediction = predict(nn, set->sample_ptrs[i]);

        if (set->sample_ptrs[i]->expected_output->data[prediction]) {
            count++;
        }

    }

    return 100 * count / ((float) set->num_samples);

}

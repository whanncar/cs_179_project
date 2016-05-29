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

    calculate_matrix_times_matrix(layer->w, layer->input, layer->s);

    /* Calculate output by filtering raw weighted sums */
    apply_filter_to_matrix_componentwise(layer->s,
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

    data_matrix *temp;

    calculate_matrix_times_matrix(next_layer->w_T,
                                next_layer->dL_ds,
                                layer->dL_ds);

    multiply_matrices_componentwise(next_layer->input,
                                   layer->dL_ds,
                                   layer->dL_ds);


    temp = new_matrix(next_layer->input->num_rows, next_layer->input->num_cols);

    calc_lin_comb_of_mats(0, next_layer->input, -1, next_layer->input, temp);

    add_constant_to_matrix(1, temp, temp);

    multiply_matrices_componentwise(temp, layer->dL_ds, layer->dL_ds);

    free_matrix(temp);
}



/*
 * UNRESOLVED
 *
 */

void compute_dL_ds_last_layer(neural_net *nn, data_matrix *expected_output) {
 
    neural_layer *last_layer;

    data_matrix *temp;

    last_layer = nn->layer_ptrs[nn->num_layers - 1];


    calc_lin_comb_of_mats(1, last_layer->output, -1, expected_output, last_layer->dL_ds);

    calc_lin_comb_of_mats(0, last_layer->dL_ds, 2, last_layer->dL_ds, last_layer->dL_ds);

    multiply_matrices_componentwise(last_layer->dL_ds, last_layer->output, last_layer->dL_ds);

    temp = new_matrix(last_layer->output->num_rows, last_layer->output->num_cols);

    calc_lin_comb_of_mats(0, last_layer->output, -1, last_layer->output, temp);

    add_constant_to_matrix(1, temp, temp);

    multiply_matrices_componentwise(last_layer->dL_ds, temp, last_layer->dL_ds);

    free_matrix(temp);
}

/*
 * UNRESOLVED
 *
 */

void compute_dL_ds_all_layers(neural_net *nn,
                              data_matrix *expected_output) {

    int i;

    compute_dL_ds_last_layer(nn, expected_output);

    for (i = nn->num_layers - 2; i >= 0; i--) {
        calculate_dL_ds_layer(nn->layer_ptrs[i], nn->layer_ptrs[i + 1]);
    }
}


/*here*/

void update_dL_dw_layer(neural_layer *layer) {

    data_matrix *temp;

    temp = new_matrix(layer->input->num_cols, layer->input->num_rows);

    compute_matrix_transpose(layer->input, temp);

    calculate_matrix_times_matrix(layer->dL_ds, temp, layer->dL_dw);

    free_matrix(temp);

}


void update_dL_dw_all_layers(neural_net *nn) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        update_dL_dw_layer(nn->layer_ptrs[i]);

    }

}



void update_w_layer(neural_layer *layer, float step) {

    calc_lin_comb_of_mats(1, layer->w, -step, layer->dL_dw, layer->w);

    compute_matrix_transpose(layer->w, layer->w_T);

}


void update_w_all_layers(neural_net *nn, float step) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        update_w_layer(nn->layer_ptrs[i], step);

    }

}


void backward_propagate_neural_net(neural_net *nn, data_matrix *expected_output, float step) {

    compute_dL_ds_all_layers(nn, expected_output);

    update_dL_dw_all_layers(nn);

    update_w_all_layers(nn, step);

}


/*
 * UNRESOLVED
 *
 */

void set_neural_net_input(neural_net *nn, data_matrix *input) {

    nn->input = input;

}




void train_neural_net(neural_net *nn, sample_set *set, float step) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    backward_propagate_neural_net(nn, set->sample_labels, step);

}




/*
 * UNRESOLVED
 *
 */

float calculate_loss(neural_net *nn, sample_set *set) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    return calculate_matrix_distance(nn->output, set->sample_labels);;

}


void predict(neural_net *nn, sample_set *set, data_matrix *predictions) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    int i, j;

    int rows, cols;

    data_matrix *output = nn->output;

    rows = output->num_rows;
    cols = output->num_cols;

    for (j = 0; j < cols; j++) {

        predictions->data[j] = 0;
        predictions->data[j + cols] = 0;

    }

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            if (output->data[i * cols + j] > predictions->data[j]) {

                predictions->data[j] = output->data[i * cols + j];
                predictions->data[cols + j] = i;

            }

        }

    }

}


float calculate_percent_predicted_correctly(neural_net *nn, sample_set *set) {

    int i;
    float count;

    data_matrix *predictions;

    predictions = new_matrix(2, set->sample_inputs->num_cols);

    predict(nn, set, predictions);

    count = 0;

    for (i = 0; i < set->sample_inputs->num_cols; i++) {

        if (set->sample_labels->data[((int) (predictions->data[predictions->num_cols + i])) * set->sample_inputs->num_cols + i]) {

            count++;

        }

    }

    free_matrix(predictions);

    return 100 * count / ((float) set->sample_inputs->num_cols);

}

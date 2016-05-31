#include "neural_net_exec.h"


void train_neural_net(neural_net *nn, sample_set *set, float step) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    backward_propagate_neural_net(nn, set->sample_labels, step);

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





void backward_propagate_neural_net(neural_net *nn, data_matrix *expected_output, float step) {

    compute_dL_ds_all_layers(nn, expected_output);

    update_dL_dw_all_layers(nn);

    update_w_all_layers(nn, step);

}





void set_neural_net_input(neural_net *nn, data_matrix *input) {

    int i, j;

    int rows, cols;

    rows = nn->input->num_rows;
    cols = nn->input->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            nn->input->data[i * cols + j] = input->data[i * cols + j];

        }

    }

}

#include "gpu_neural_net_exec.h"


void gpu_train_neural_net(neural_net *nn, sample_set *set, float step) {

    gpu_forward_propagate_neural_net(nn);
/*
    gpu_backward_propagate_neural_net(nn, set->sample_labels, step);
*/
}




/*
 * forward_propagate_neural_net: Forward propagates the given neural net
 *                               using the given filter
 *
 * arguments: nn: The neural net to be forward propagated
 *
 */

void gpu_forward_propagate_neural_net(neural_net *nn) {

    int i;

    /* Propagate each layer */
    for (i = 0; i < nn->num_layers; i++) {
        gpu_forward_propagate_layer(nn->layer_ptrs[i]);
    }

}





void gpu_backward_propagate_neural_net(neural_net *nn, data_matrix *expected_output, float step) {

    gpu_compute_dL_ds_all_layers(nn, expected_output);

    gpu_update_dL_dw_all_layers(nn);

    gpu_update_w_all_layers(nn, step);

}






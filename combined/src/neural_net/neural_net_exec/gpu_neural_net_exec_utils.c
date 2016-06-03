#include "gpu_neural_net_exec_utils.h"


void gpu_calculate_dL_ds_layer(neural_layer *, neural_layer *);
void gpu_compute_dL_ds_last_layer(neural_net *, data_matrix *);
void gpu_update_dL_dw_layer(neural_layer *);
void gpu_update_w_layer(neural_layer *, float);



/* Important functions */


void gpu_compute_dL_ds_all_layers(neural_net *nn,
                                  data_matrix *expected_output) {

    int i;

    gpu_compute_dL_ds_last_layer(nn, expected_output);

    for (i = nn->num_layers - 2; i >= 0; i--) {
        gpu_calculate_dL_ds_layer(nn->layer_ptrs[i], nn->layer_ptrs[i + 1]);
    }
}


void gpu_update_dL_dw_all_layers(neural_net *nn) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        gpu_update_dL_dw_layer(nn->layer_ptrs[i]);

    }

}



void gpu_update_w_all_layers(neural_net *nn, float step) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        gpu_update_w_layer(nn->layer_ptrs[i], step);

    }

}







void gpu_forward_propagate_layer(neural_layer *layer) {

    gpu_calculate_matrix_times_matrix(layer->w, layer->input, layer->s);

    /* Calculate output by filtering raw weighted sums */
    gpu_apply_sigmoid_to_matrix_componentwise(layer->s,
                                             layer->output);

}


/* Helper functions */


void gpu_calculate_dL_ds_layer(neural_layer *layer,
                               neural_layer *next_layer) {

    data_matrix *temp;

    gpu_calculate_matrix_times_matrix(next_layer->w_T,
                                      next_layer->dL_ds,
                                      layer->dL_ds);

    gpu_multiply_matrices_componentwise(next_layer->input,
                                        layer->dL_ds,
                                        layer->dL_ds);


    temp = gpu_new_matrix(next_layer->input->num_rows, next_layer->input->num_cols);

    gpu_calc_lin_comb_of_mats(0.0, next_layer->input, -1.0, next_layer->input, temp);

    gpu_add_constant_to_matrix(1.0, temp, temp);

    gpu_multiply_matrices_componentwise(temp, layer->dL_ds, layer->dL_ds);

    gpu_free_matrix(temp);
}



void gpu_compute_dL_ds_last_layer(neural_net *nn, data_matrix *expected_output) {
 
    neural_layer *last_layer;

    data_matrix *temp;

    last_layer = nn->layer_ptrs[nn->num_layers - 1];


    gpu_calc_lin_comb_of_mats(1.0, last_layer->output, -1.0, expected_output, last_layer->dL_ds);

    gpu_calc_lin_comb_of_mats(0.0, last_layer->dL_ds, 2.0, last_layer->dL_ds, last_layer->dL_ds);

    gpu_multiply_matrices_componentwise(last_layer->dL_ds, last_layer->output, last_layer->dL_ds);

    temp = gpu_new_matrix(last_layer->output->num_rows, last_layer->output->num_cols);

    gpu_calc_lin_comb_of_mats(0.0, last_layer->output, -1.0, last_layer->output, temp);

    gpu_add_constant_to_matrix(1.0, temp, temp);

    gpu_multiply_matrices_componentwise(last_layer->dL_ds, temp, last_layer->dL_ds);

    gpu_free_matrix(temp);
}





void gpu_update_dL_dw_layer(neural_layer *layer) {

    data_matrix *temp;

    temp = gpu_new_matrix(layer->input->num_cols, layer->input->num_rows);

    gpu_compute_matrix_transpose(layer->input, temp);

    gpu_calculate_matrix_times_matrix(layer->dL_ds, temp, layer->dL_dw);

    gpu_free_matrix(temp);

}





void gpu_update_w_layer(neural_layer *layer, float step) {

    gpu_calc_lin_comb_of_mats(1, layer->w, -step, layer->dL_dw, layer->w);

    gpu_compute_matrix_transpose(layer->w, layer->w_T);

}

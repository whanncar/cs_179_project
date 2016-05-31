#include "neural_net_exec_utils.h"


void calculate_dL_ds_layer(neural_layer *, neural_layer *);
void compute_dL_ds_last_layer(neural_net *, data_matrix *);
void update_dL_dw_layer(neural_layer *);
void update_w_layer(neural_layer *, float);



/* Important functions */


void compute_dL_ds_all_layers(neural_net *nn,
                              data_matrix *expected_output) {

    int i;

    compute_dL_ds_last_layer(nn, expected_output);

    for (i = nn->num_layers - 2; i >= 0; i--) {
        calculate_dL_ds_layer(nn->layer_ptrs[i], nn->layer_ptrs[i + 1]);
    }
}


void update_dL_dw_all_layers(neural_net *nn) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        update_dL_dw_layer(nn->layer_ptrs[i]);

    }

}



void update_w_all_layers(neural_net *nn, float step) {

    int i;

    for (i = 0; i < nn->num_layers; i++) {

        update_w_layer(nn->layer_ptrs[i], step);

    }

}




float sigmoid_filter(float x) {

    return 1 / (1 + expf(-x));

}


void forward_propagate_layer(neural_layer *layer) {

    calculate_matrix_times_matrix(layer->w, layer->input, layer->s);

    /* Calculate output by filtering raw weighted sums */
    apply_filter_to_matrix_componentwise(layer->s,
                                         &sigmoid_filter,
                                         layer->output);

}


/* Helper functions */


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

    calc_lin_comb_of_mats(0.0, next_layer->input, -1.0, next_layer->input, temp);

    add_constant_to_matrix(1.0, temp, temp);

    multiply_matrices_componentwise(temp, layer->dL_ds, layer->dL_ds);

    free_matrix(temp);
}



void compute_dL_ds_last_layer(neural_net *nn, data_matrix *expected_output) {
 
    neural_layer *last_layer;

    data_matrix *temp;

    last_layer = nn->layer_ptrs[nn->num_layers - 1];


    calc_lin_comb_of_mats(1.0, last_layer->output, -1.0, expected_output, last_layer->dL_ds);

    calc_lin_comb_of_mats(0.0, last_layer->dL_ds, 2.0, last_layer->dL_ds, last_layer->dL_ds);

    multiply_matrices_componentwise(last_layer->dL_ds, last_layer->output, last_layer->dL_ds);

    temp = new_matrix(last_layer->output->num_rows, last_layer->output->num_cols);

    calc_lin_comb_of_mats(0.0, last_layer->output, -1.0, last_layer->output, temp);

    add_constant_to_matrix(1.0, temp, temp);

    multiply_matrices_componentwise(last_layer->dL_ds, temp, last_layer->dL_ds);

    free_matrix(temp);
}





void update_dL_dw_layer(neural_layer *layer) {

    data_matrix *temp;

    temp = new_matrix(layer->input->num_cols, layer->input->num_rows);

    compute_matrix_transpose(layer->input, temp);

    calculate_matrix_times_matrix(layer->dL_ds, temp, layer->dL_dw);

    free_matrix(temp);

}





void update_w_layer(neural_layer *layer, float step) {

    calc_lin_comb_of_mats(1, layer->w, -step, layer->dL_dw, layer->w);

    compute_matrix_transpose(layer->w, layer->w_T);

}

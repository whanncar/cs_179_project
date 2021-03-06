
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "../utils/utils.h"

/* Structs */

typedef struct {

    data_vector *input;

    data_matrix *w;
    data_matrix *w_T;

    data_vector *s;

    data_vector *output;

    data_vector *dL_ds_local;
    data_matrix *dL_dw;

} neural_layer;



typedef struct {

    data_vector *input;
    data_vector *output;

    int num_layers;

    neural_layer **layer_ptrs;

} neural_net;



typedef struct {

    data_vector *input;
    data_vector *expected_output;

} sample;



typedef struct {

    int num_samples;

    sample **sample_ptrs;

} sample_set;

/* Functions */

void train_neural_net(neural_net *, sample_set *, float step);

float calculate_loss(neural_net *, sample_set *);

float calculate_percent_predicted_correctly(neural_net *, sample_set *);

#endif /* NEURAL_NET_H */

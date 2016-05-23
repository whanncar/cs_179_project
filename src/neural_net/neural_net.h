#include "../utils/utils.h"

/* Structs */

typedef struct {

    data_vector *input;

    data_matrix *w;
    data_matrix *w_T;

    data_vector *r;

    data_vector *t;

    data_vector *s;

    data_vector *output;

    data_vector *dL_ds;

} neural_layer;



typedef struct {

    data_vector *input;
    data_vector *output;

    int num_layers;

    neural_layer **layer_ptrs;

} neural_net;



/* Functions */

void forward_propagate_neural_net(neural_net *);

void backward_propagate_neural_net(neural_net *,
                                   data_vector *expected_output,
                                   float step);

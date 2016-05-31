
#ifndef NEURAL_LAYER_CUH
#define NEURAL_LAYER_CUH

#include "../utils/utils.h"

/* Structs */

typedef struct {

    data_matrix *input;

    data_matrix *w;
    data_matrix *w_T;

    data_matrix *s;

    data_matrix *output;

    data_matrix *dL_ds;
    data_matrix *dL_dw;

} neural_layer;



/* Functions */

neural_layer *gpu_new_neural_layer(int input_length,
                                   int num_weights,
                                   int num_inputs);

void gpu_free_neural_layer(neural_layer *);

#endif /* NEURAL_LAYER_CUH */

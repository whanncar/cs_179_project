
#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

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

#endif /* NEURAL_LAYER_H */


#ifndef NEURAL_NET_CUH
#define NEURAL_NET_CUH

#include <stdlib.h>
#include "../utils/utils.h"
#include "neural_layer.h"

/* Structs */

typedef struct {

    data_matrix *input;
    data_matrix *output;

    int num_layers;

    neural_layer **layer_ptrs;

} neural_net;



typedef struct {

    data_matrix *sample_inputs;
    data_matrix *sample_labels;

} sample_set;



/* Functions */

neural_net *gpu_new_neural_net(int num_layers, int num_inputs,
                               int input_size, int output_size,
                               int *layer_weight_specs);

void gpu_free_neural_net(neural_net *);

#endif /* NEURAL_NET_CUH */

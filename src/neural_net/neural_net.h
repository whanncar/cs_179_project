#include "../utils/utils.h"



typedef struct {

    data_vector *input_data;
    data_vector *output_data;

    data_matrix *weights;

} neural_layer;



typedef struct {

    data_vector *input_data;
    data_vector *output_data;

    int num_layers;

    neural_layer **layer_ptrs;

} neural_net;



void forward_propagate_neural_net(neural_net *, float (*filter)(float));
void free_neural_net(neural_net *);

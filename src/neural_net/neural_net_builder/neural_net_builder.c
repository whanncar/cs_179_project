#include "neural_net_builder.h"
#include "../neural_net.h"


neural_net *new_neural_net(neural_net_config *config) {

    neural_net *new_net;
    int i;
    layer_config_node *current_layer_config;
    data_vector **data_vector_ptrs;

    /* Allocate space for new neural net */
    new_net = (neural_net *) malloc(sizeof(neural_net));

    /* Initialize new neural net */
    new_net->num_layers = config->num_layers;
    new_net->layer_ptrs = (neural_layer **)
        malloc((new_net->num_layers) * sizeof(neural_layer *));

    for (i = 0, current_layer_config = config->first;
         i < new_net->num_layers;
         i++, current_layer_config = current_layer_config->next) {

        (new_net->layer_ptrs)[i] = (neural_layer *)
                                      malloc(sizeof(neural_layer));

        ((new_net->layer_ptrs)[i])->weights =
            new_data_matrix(current_layer_config->number_of_weights,
                            current_layer_config->input_vector_size);

    }

    /* Finish writing me UNRESOLVED */

}

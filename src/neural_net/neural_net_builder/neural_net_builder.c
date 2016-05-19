#include "neural_net_builder.h"
#include "../neural_net.h"
#include <stdio.h>


/*
 * new_neural_net: Creates and allocates memory for a new
 *                 neural net based on the given neural
 *                 net configuration
 *
 * arguments: config: Configuration for neural net structure
 *
 * return value: Pointer to new neural net
 *
 */

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

    /* For each neural layer to be built */
    for (i = 0, current_layer_config = config->first;
         i < new_net->num_layers;
         i++, current_layer_config = current_layer_config->next) {

        /* Allocate space for the new neural layer */
        (new_net->layer_ptrs)[i] = (neural_layer *)
                                      malloc(sizeof(neural_layer));

        /* Allocate space for the new neural layer's weights */
        ((new_net->layer_ptrs)[i])->weights =
            new_data_matrix(current_layer_config->number_of_weights,
                            current_layer_config->input_vector_size);
    }



    /* Allocate temporary array to hold data vector pointers */
    data_vector_ptrs = (data_vector **) 
          malloc((new_net->num_layers + 1) * sizeof(data_vector *));

    /* For each neural layer */
    for (i = 0, current_layer_config = config->first;
         i < new_net->num_layers;
         i++, current_layer_config = current_layer_config->next) {

        /* Make new data vector of appropriate size */
        data_vector_ptrs[i] = 
            new_vector(current_layer_config->input_vector_size);
    }

    /* Make output data vector */
    data_vector_ptrs[i] = new_vector(config->last->number_of_weights);



    /* For each layer */
    for (i = 0; i < new_net->num_layers; i++) {
        /* Set input and output data pointers */
        ((new_net->layer_ptrs)[i])->input_data = data_vector_ptrs[i];
        ((new_net->layer_ptrs)[i])->output_data = data_vector_ptrs[i+1];
    }



    /* Set neural net input and output data pointers */
    new_net->input_data = data_vector_ptrs[0];
    new_net->output_data = data_vector_ptrs[i];



    /* Clean up */
    free(data_vector_ptrs);

    return new_net;

}




neural_net_config *load_config(char *filename) {

    

}


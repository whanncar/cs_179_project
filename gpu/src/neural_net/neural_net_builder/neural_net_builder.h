#include "../neural_net.h"



typedef struct _layer_config_linked_list_node {

    int input_vector_size;
    int number_of_weights; 

} layer_config_node;



typedef struct _layer_config_linked_list {

    int num_layers;

    layer_config_node *first;
    layer_config_node *last;

} neural_net_config;



neural_net *new_neural_net(neural_net_config *);
neural_net_config *load_config(char *);

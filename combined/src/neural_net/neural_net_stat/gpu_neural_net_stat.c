#include "gpu_neural_net_stat.h"





float gpu_calculate_loss(neural_net *nn, sample_set *set) {

    gpu_forward_propagate_neural_net(nn);

    return gpu_calculate_matrix_distance(nn->output, set->sample_labels);

}



void gpu_predict(neural_net *nn, sample_set *set, data_matrix *predictions) {

/* TODO */

}


float gpu_calculate_percent_predicted_correctly(neural_net *nn, sample_set *set) {

/* TODO */

}

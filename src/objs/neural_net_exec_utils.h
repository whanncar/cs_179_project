
#ifndef NEURAL_NET_EXEC_UTILS_H
#define NEURAL_NET_EXEC_UTILS_H

#include "../../utils/utils.h"
#include "../neural_net.h"
#include <math.h>


void compute_dL_ds_all_layers(neural_net *, data_matrix *expected_output);

void update_dL_dw_all_layers(neural_net *);

void update_w_all_layers(neural_net *, float step);

void forward_propagate_layer(neural_layer *);

#endif /* NEURAL_NET_EXEC_UTILS_H */

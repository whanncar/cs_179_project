
#ifndef GPU_NEURAL_NET_EXEC_UTILS_H
#define GPU_NEURAL_NET_EXEC_UTILS_H

#include "../../utils/gpu_utils.h"
#include "../gpu_neural_net.h"
#include <math.h>


void gpu_compute_dL_ds_all_layers(neural_net *, data_matrix *expected_output);

void gpu_update_dL_dw_all_layers(neural_net *);

void gpu_update_w_all_layers(neural_net *, float step);

void gpu_forward_propagate_layer(neural_layer *);

#endif /* GPU_NEURAL_NET_EXEC_UTILS_H */
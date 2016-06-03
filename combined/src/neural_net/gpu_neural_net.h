
#ifndef GPU_NEURAL_NET_H
#define GPU_NEURAL_NET_H

#include <stdlib.h>
#include "../utils/gpu_utils.h"
#include "gpu_neural_layer.h"
#include "neural_net.h"

/* Functions */

neural_net *gpu_new_neural_net(int num_layers, int num_inputs,
                               int input_size, int output_size,
                               int *layer_weight_specs);

void gpu_free_neural_net(neural_net *);

void copy_neural_net_to_gpu(neural_net *nn, neural_net *nn_dev);

#endif /* GPU_NEURAL_NET_H */
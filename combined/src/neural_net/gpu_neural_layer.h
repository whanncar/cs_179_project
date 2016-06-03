
#ifndef GPU_NEURAL_LAYER_H
#define GPU_NEURAL_LAYER_H

#include "../utils/gpu_utils.h"
#include "neural_layer.h"

/* Functions */

neural_layer *gpu_new_neural_layer(int input_length,
                                   int num_weights,
                                   int num_inputs);

void gpu_free_neural_layer(neural_layer *);

void copy_neural_layer_to_gpu(neural_layer *layer, neural_layer *layer_dev);

#endif /* GPU_NEURAL_LAYER_H */

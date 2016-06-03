
#ifndef GPU_NEURAL_NET_EXEC_H
#define GPU_NEURAL_NET_EXEC_H

#include "../gpu_neural_net.h"
#include "gpu_neural_net_exec_utils.h"
#include "../../utils/gpu_utils.h"

void gpu_forward_propagate_neural_net(neural_net *);
void gpu_backward_propagate_neural_net(neural_net *, data_matrix *, float);

void gpu_train_neural_net(neural_net *nn, sample_set *set, float step);

#endif /* GPU_NEURAL_NET_EXEC_H */


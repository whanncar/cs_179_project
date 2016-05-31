
#ifndef NEURAL_NET_EXEC_H
#define NEURAL_NET_EXEC_H

#include "../neural_net.h"
#include "neural_net_exec_utils.h"
#include "../../utils/utils.h"

void gpu_forward_propagate_neural_net(neural_net *);
void gpu_backward_propagate_neural_net(neural_net *, data_matrix *, float);

void gpu_train_neural_net(neural_net *nn, sample_set *set, float step);

#endif /* NEURAL_NET_EXEC_H */


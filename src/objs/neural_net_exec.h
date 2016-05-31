
#ifndef NEURAL_NET_EXEC_H
#define NEURAL_NET_EXEC_H

#include "../neural_net.h"
#include "neural_net_exec_utils.h"
#include "../../utils/utils.h"

void forward_propagate_neural_net(neural_net *);
void backward_propagate_neural_net(neural_net *, data_matrix *, float);
void set_neural_net_input(neural_net *, data_matrix *);

#endif /* NEURAL_NET_EXEC_H */


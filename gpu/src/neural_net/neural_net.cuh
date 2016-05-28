#include "../../../src/neural_net/neural_net.h"
#include "../utils/utils.cuh"

#ifndef NEURAL_NET_CUH
#define NEURAL_NET_CUH

void train_neural_net(neural_net *, sample_set *, float step);

float calculate_loss(neural_net *, sample_set *);

float calculate_percent_predicted_correctly(neural_net *, sample_set *);

#endif /* NEURAL_NET_CUH */

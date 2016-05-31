
#ifndef NEURAL_NET_STAT_H
#define NEURAL_NET_STAT_H

#include "../../utils/utils.h"
#include "../neural_net.h"
#include "../neural_net_exec/neural_net_exec.h"

float gpu_calculate_loss(neural_net *, sample_set *);

void gpu_predict(neural_net *, sample_set *, data_matrix *predictions);

float gpu_calculate_percent_predicted_correctly(neural_net *, sample_set *);

#endif /* NEURAL_NET_STAT_H */

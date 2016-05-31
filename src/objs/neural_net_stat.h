
#ifndef NEURAL_NET_STAT_H
#define NEURAL_NET_STAT_H

#include "../../utils/utils.h"
#include "../neural_net.h"
#include "../neural_net_exec/neural_net_exec.h"

float calculate_loss(neural_net *, sample_set *);

void predict(neural_net *, sample_set *, data_matrix *predictions);

float calculate_percent_predicted_correctly(neural_net *, sample_set *);

#endif /* NEURAL_NET_STAT_H */

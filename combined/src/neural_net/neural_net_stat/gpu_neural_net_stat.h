
#ifndef GPU_NEURAL_NET_STAT_H
#define GPU_NEURAL_NET_STAT_H

#include "../../utils/gpu_utils.h"
#include "../gpu_neural_net.h"
#include "../neural_net_exec/gpu_neural_net_exec.h"

float gpu_calculate_loss(neural_net *, sample_set *);

void gpu_predict(neural_net *, sample_set *, data_matrix *predictions);

float gpu_calculate_percent_predicted_correctly(neural_net *, sample_set *);

#endif /* GPU_NEURAL_NET_STAT_H */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "neural_net/neural_net.h"
#include "neural_net/neural_net_exec/neural_net_exec.h"
#include "neural_net/neural_net_stat/neural_net_stat.h"
#include "io_utils/fileio.h"

#include "neural_net/gpu_neural_net.h"
#include "neural_net/neural_net_exec/gpu_neural_net_exec.h"
#include "neural_net/neural_net_stat/gpu_neural_net_stat.h"


void initialize_neural_net(int, char **);
void gpu_initialize_neural_net(int, char **);
void initialize_samples(int, char **);
void gpu_initialize_samples();

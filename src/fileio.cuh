
#ifndef FILEIO_H
#define FILEIO_H

#include "neural_net.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"



sample_set *get_samples_from_file(char *filepath,
                                  int num_samples,
                                  int sample_length);

#endif /* FILEIO_H */

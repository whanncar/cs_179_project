
#ifndef FILEIO_H
#define FILEIO_H

#include "../neural_net/neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include "../utils/utils.h"



sample_set *get_samples_from_file(char *filepath,
                                  int num_samples,
                                  int sample_length);

#endif /* FILEIO_H */

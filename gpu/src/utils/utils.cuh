#include "../../../src/utils/utils.h"

#ifndef UTILS_CUH
#define UTILS_CUH

data_vector *gpu_new_vector(int size);

data_matrix *gpu_new_matrix(int num_rows, int num_cols);

void gpu_multiply_vector_by_constant(data_vector *, float,
                                     data_vector *result)

void gpu_add_vectors(data_vector *, data_vector *, data_vector *result);

void gpu_compute_additive_inverse_of_vector(data_vector *, data_vector *result);

void gpu_add_constant_componentwise_to_vector(data_vector *v, float c,
                                              data_vector *result);

void gpu_multiply_vectors_componentwise(data_vector *, data_vector *,
                                        data_vector *result);



#endif /* UTILS_CUH */

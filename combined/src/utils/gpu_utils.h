
#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "gpu_utils_cuda.h"

/* Functions */

data_matrix *gpu_new_matrix(int num_rows, int num_cols);

void gpu_calculate_matrix_times_matrix(data_matrix *m1,
                                       data_matrix *m2,
                                       data_matrix *result);

void gpu_calc_lin_comb_of_mats(float a, data_matrix *m1,
                               float b, data_matrix *m2,
                               data_matrix *result);

void gpu_add_constant_to_matrix(float c, data_matrix *m, data_matrix *result);

void gpu_multiply_matrices_componentwise(data_matrix *m1, data_matrix *m2, data_matrix *result);

void gpu_compute_matrix_transpose(data_matrix *m, data_matrix *result);

void gpu_apply_sigmoid_to_matrix_componentwise(data_matrix *m,
                                               data_matrix *result);

float gpu_calculate_matrix_distance(data_matrix *m1, data_matrix *m2);

void gpu_free_matrix(data_matrix *);

#endif /* GPU_UTILS_H */

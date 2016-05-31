
#ifndef UTILS_H
#define UTILS_H

/* Structs */

typedef struct {

    int num_rows;
    int num_cols;

    float *data;

    int stride;

} data_matrix;



/* Functions */

data_matrix *new_matrix(int num_rows, int num_cols);

void calculate_matrix_times_matrix(data_matrix *m1,
                                   data_matrix *m2,
                                   data_matrix *result);

void free_matrix(data_matrix *m);

void calc_lin_comb_of_mats(float a, data_matrix *m1,
                           float b, data_matrix *m2,
                           data_matrix *result);

void add_constant_to_matrix(float c, data_matrix *m, data_matrix *result);

void multiply_matrices_componentwise(data_matrix *m1, data_matrix *m2, data_matrix *result);

void compute_matrix_transpose(data_matrix *m, data_matrix *result);

void apply_filter_to_matrix_componentwise(data_matrix *m,
                                          float (*filter)(float),
                                          data_matrix *result);

void fill_matrix_rand(data_matrix *m, float min, float max);

float calculate_matrix_distance(data_matrix *m1, data_matrix *m2);

#endif /* UTILS_H */

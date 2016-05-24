
#ifndef UTILS_H
#define UTILS_H

/* Structs */

typedef struct {

    int size;

    float *data;

} data_vector;



typedef struct {

    int num_rows;
    int num_cols;

    float *data;

} data_matrix;



/* Functions */

data_vector *new_vector(int size);

data_matrix *new_matrix(int num_rows, int num_cols);

void calculate_matrix_times_vector(data_matrix *,
                                   data_vector *,
                                   data_vector *result);

void multiply_vector_by_constant(data_vector *, float, data_vector *result);

void add_vectors(data_vector *, data_vector *, data_vector *result)

void compute_additive_inverse_of_vector(data_vector *, data_vector *result);

void add_constant_componentwise_to_vector(data_vector *, float,
                                          data_vector *result);

void multiply_vectors_componentwise(data_vector *, data_vector *,
                                    data_vector *result);

void compute_matrix_transpose(data_matrix *, data_matrix *result);

void apply_filter_to_vector_componentwise(data_vector *,
                                          float (*filter)(float),
                                          data_vector *result) {


void fill_matrix_rand(data_matrix *, float min, float max);

#endif /* UTILS_H */


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

#endif /* UTILS_H */

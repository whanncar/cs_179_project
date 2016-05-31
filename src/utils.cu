#include <stdlib.h>
#include <assert.h>
#include "utils.cuh"



/* 
 * new_matrix: Creates a new data matrix with the given
 *                  dimensions
 *
 * arguments: num_rows: The number of rows in the new data matrix
 *            num_cols: The number of columns in the new data matrix
 *
 * return value: Pointer to the newly allocated data matrix
 *
 */

data_matrix *new_matrix(int num_rows, int num_cols) {

    data_matrix *new_matrix;

    /* Allocate space for new matrix */
    new_matrix = (data_matrix *) malloc(sizeof(data_matrix));

    /* Initialize new matrix's structure values */
    new_matrix->num_rows = num_rows;
    new_matrix->num_cols = num_cols;

    /* Allocate space for new matrix's data */
    new_matrix->data = (float *) malloc(num_rows * num_cols * sizeof(float));

    /* Set the matrix's stride */
    new_matrix->stride = num_cols;

    return new_matrix;
}


void free_matrix(data_matrix *m) {

    free(m->data);
    free(m);

}


void calculate_matrix_times_matrix(data_matrix *m1,
                                   data_matrix *m2,
                                   data_matrix *result) {

    int i, j, k;

    int m1_rows, m1_cols, m2_cols;

    assert(m1->num_cols == m2->num_rows);
    assert(m1->num_rows == result->num_rows);
    assert(m2->num_cols == result->num_cols);


    m1_rows = m1->num_rows;
    m1_cols = m1->num_cols;
    m2_cols = m2->num_cols;

    for (i = 0; i < m1_rows; i++) {

        for (j = 0; j < m2_cols; j++) {

            result->data[i * m2_cols + j] = 0;

            for (k = 0; k < m1_cols; k++) {

                result->data[i * m2_cols + j] +=
                    m1->data[i * m1_cols + k] * m2->data[k * m2_cols + j];

            }

        }

    }

}


void calc_lin_comb_of_mats(float a, data_matrix *m1,
                           float b, data_matrix *m2,
                           data_matrix *result) {

    int i, j;

    int rows, cols;

    assert(m1->num_rows == m2->num_rows);
    assert(m1->num_cols == m2->num_cols);

    rows = m1->num_rows;
    cols = m1->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            result->data[i * cols + j] = a * m1->data[i * cols + j] + 
                                         b * m2->data[i * cols + j];

        }

    }

}


void add_constant_to_matrix(float c, data_matrix *m, data_matrix *result) {

    int i, j;

    int rows, cols;

    rows = m->num_rows;
    cols = m->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            result->data[i * cols + j] = c + m->data[i * cols + j];

        }

    }

}


void multiply_matrices_componentwise(data_matrix *m1, data_matrix *m2, data_matrix *result) {

    int i, j;

    int rows, cols;

    assert(m1->num_rows == m2->num_rows);
    assert(m1->num_cols == m2->num_cols);

    rows = m1->num_rows;
    cols = m1->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            result->data[i * cols + j] = (m1->data[i * cols + j]) * (m2->data[i * cols + j]);

        }

    }

}


/*
 * UNRESOLVED
 *
 */

void compute_matrix_transpose(data_matrix *m, data_matrix *result) {

    int i, j;

    assert(m->num_rows == result->num_cols);
    assert(m->num_cols == result->num_rows);

    for (i = 0; i < m->num_rows; i++) {

        for (j = 0; j < m->num_cols; j++) {

            result->data[j * result->num_cols + i] = m->data[i * m->num_cols + j];

        }

    }

}



/*
 * UNRESOLVED
 *
 */

void apply_filter_to_matrix_componentwise(data_matrix *m,
                                          float (*filter)(float),
                                          data_matrix *result) {

    int i, j;

    int rows, cols;

    rows = m->num_rows;
    cols = m->num_cols;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            result->data[i * cols + j] = filter(m->data[i * cols + j]);

        }

    }

}



/*
 * UNRESOLVED
 *
 */

void fill_matrix_rand(data_matrix *m, float min, float max) {

    int i, j;
    float r;

    for (i = 0; i < m->num_rows; i++) {

        for (j = 0; j < m->num_cols; j++) {

            r = ((float) rand()) / ((float) RAND_MAX);

            r *= (max - min);

            r += min;

            m->data[i * m->num_cols + j] = r;

        }

    }

}



float calculate_matrix_distance(data_matrix *m1, data_matrix *m2) {

    int i, j;
    int rows, cols;
    float result, diff;

    assert(m1->num_rows == m2->num_rows);
    assert(m1->num_cols == m2->num_cols);

    rows = m1->num_rows;
    cols = m1->num_cols;

    result = 0;

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            diff = m1->data[i * cols + j] - m2->data[i * cols + j];

            result += diff * diff;

        }

    }

    return result;

}

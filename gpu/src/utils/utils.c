#include "utils.h"



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

data_matrix *gpu_new_matrix(int num_rows, int num_cols) {

    data_matrix *new_matrix;

    /* Allocate space for new matrix */
    new_matrix = (data_matrix *) malloc(sizeof(data_matrix));

    /* Initialize new matrix's structure values */
    new_matrix->num_rows = num_rows;
    new_matrix->num_cols = num_cols;

    cudaMalloc((void **) &(new_matrix->data),
               num_rows * num_cols * sizeof(float));

    /* Set the matrix's stride */
    new_matrix->stride = num_cols;

    return new_matrix;
}


void gpu_free_matrix(data_matrix *m) {

    cudaFree(m->data);
    free(m);

}


void gpu_calculate_matrix_times_matrix(data_matrix *m1,
                                       data_matrix *m2,
                                       data_matrix *result) {

/* TODO */

}


void gpu_calc_lin_comb_of_mats(float a, data_matrix *m1,
                               float b, data_matrix *m2,
                               data_matrix *result) {

callLinCombOfVectors(a, m1->data, b, float m2->data,
                     m1->num_rows * m1->num_cols, result->data);

}


void gpu_add_constant_to_matrix(float c, data_matrix *m, data_matrix *result) {

callAddConstantToVector(c, m->data, m->num_rows * m->num_cols, result->data);

}


void gpu_multiply_matrices_componentwise(data_matrix *m1, data_matrix *m2, data_matrix *result) {

callMultVectsCompwise(m1->data, m2->data, m1->num_rows * m2->num_rows, result->data);

}


void gpu_compute_matrix_transpose(data_matrix *m, data_matrix *result) {

callMatrixTranspose(m->data, result->data, m->num_rows, m->num_cols);

}


void gpu_apply_sigmoid_to_matrix_componentwise(data_matrix *m,
                                               data_matrix *result) {

callApplySigmoidToVector(m->data, m->num_rows * m->num_cols, result->data);

}



float gpu_calculate_matrix_distance(data_matrix *m1, data_matrix *m2) {

return callCalcVectDist(m1->data, m2->data, m1->num_rows * m1->num_cols);

}

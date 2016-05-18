#include <stdlib.h>
#include <assert.h>
#include "utils.h"



data_vector *get_row(data_matrix *, int);



/*
 * new_vector: Creates a new data vector with
 *                  space for an array of the given
 *                  size
 *
 * arguments: size: Size of the array to be stored
 * 
 * return value: Pointer to the newly allocated data vector
 *
 */

data_vector *new_vector(int size) {

    assert(size >= 0);

    data_vector *new_vector;

    /* Allocate space for the new input vector */
    new_vector = (data_vector *) malloc(sizeof(data_vector) +
                                         size * sizeof(float));

    /* Set new input vector's size */
    new_vector->size = size;

    return new_vector;
}



/*
 * get_vector_data: Returns pointer to the array associated
 *                  with the given data vector
 *
 * arguments: v: Data vector whose array is to be returned
 *
 * return value: Array associated with the given data vector
 *
 */

float *get_vector_data(data_vector *v) {

    return (float *) (v + 1);

}



/*
 * dot_product: Computes the dot product of the given
 *              data vectors
 *
 * arguments: v1: First vector
 *            v2: Second vector
 *
 * return value: Dot product of v1 and v2
 *
 */

float dot_product(data_vector *v1, data_vector *v2) {

    int length, i;
    float result;

    float *v1_data;
    float *v2_data;

    assert(v1->size == v2->size);

    length = v1->size;

    v1_data = get_vector_data(v1);
    v2_data = get_vector_data(v2);

    result = 0;

    length = v1->size;

    for (i = 0; i < length; i++) {
        result += v1_data[i] * v2_data[i];
    }

    return result;
}



/* 
 * new_data_matrix: Creates a new data matrix with the given
 *                  dimensions
 *
 * arguments: num_rows: The number of rows in the new data matrix
 *            num_cols: The number of columns in the new data matrix
 *
 * return value: Pointer to the newly allocated data matrix
 *
 */

data_matrix *new_data_matrix(int num_rows, int num_cols) {

    data_matrix *new_matrix;
    int total_size;
    int i;
    data_vector *current_row;

    /* Calculate total size of new data matrix */
    total_size = sizeof(data_matrix);
    total_size += (sizeof(data_vector) + num_cols * sizeof(float)) * num_rows;

    /* Allocate space for new data matrix */
    new_matrix = (data_matrix *) malloc(total_size);

    /* Initialize matrix structure values */
    new_matrix->num_rows = num_rows;
    new_matrix->num_cols = num_cols;

    /* Initialize rows of new data matrix */
    for (i = 0; i < num_rows; i++) {
        current_row = get_row(new_matrix, i);
        current_row->size = num_cols;
    }

    return new_matrix;
}



/*
 * get_row: Retrieves the data vector corresponding
 *          to the given row from the given matrix
 *
 * arguments: m: The matrix to retrieve the row from
 *            row: The index of the row to be retrieved
 *
 * return value: Pointer to the data vector corresponding
 *               to the desired row of the given matrix
 *
 */

data_vector *get_row(data_matrix *m, int row) {

    char *row_vector;

    row_vector = (char *) (m + 1);

    row_vector += (sizeof(data_vector) + (m->num_cols) * sizeof(float)) * row;

    return (data_vector *) row_vector;

}



/*
 * calculate_matrix_times_vector: Multiplies the matrix m by the vector
 *                                v and stores the product in the vector
 *                                result
 *
 * arguments: m: Matrix for multiplication
 *            v: Vector for multiplication
 *            result: Vector for storing result
 *
 */

void calculate_matrix_times_vector(data_matrix *m,
                                   data_vector *v,
                                   data_vector *result) {

    data_vector *current_row;
    float *result_data;
    int i;

    result_data = get_vector_data(result);

    for (i = 0, current_row = get_row(m, 0);
         i < m->num_rows;
         i++, current_row = get_row(m, i)) {

        result_data[i] = dot_product(current_row, v);

    }

}



/*
 * filter_vector: Computes a filtered value associated
 *                with entry of a given vector and stores
 *                the results in a given vector
 *
 * arguments: unfiltered_v: The vector to apply the filter to
 *            filtered_v: The vector to store the filtered
 *                        values in
 *            filter: The function for filtering the values
 *
 */

void filter_vector(data_vector *unfiltered_v,
                   data_vector *filtered_v,
                   float (*filter)(float)) {

    float *unfiltered_arr;
    float *filtered_arr;
    int i;
    int length;

    assert(unfiltered_v->size == filtered_v->size);

    /* Get vectors */
    unfiltered_arr = get_vector_data(unfiltered_v);
    filtered_arr = get_vector_data(filtered_v);

    /* Get length of vectors */
    length = unfiltered_v->size;

    /* Filter */
    for (i = 0; i < length; i++) {
        filtered_arr[i] = filter(unfiltered_arr[i]);
    }

}











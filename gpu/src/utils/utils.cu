#include <stdlib.h>
#include <assert.h>
#include "utils.h"

/* Change all descriptions to say "on GPU" UNRESOLVED */

/*
 * new_vector: Creates a new data vector with
 *             space for an array of the given
 *             size
 *
 * arguments: size: Size of the array to be stored
 * 
 * return value: Pointer to the newly allocated data vector
 *
 */

data_vector *gpu_new_vector(int size) {

    data_vector *new_vector;

    /* Allocate space for the new vector */
    new_vector = (data_vector *) malloc(sizeof(data_vector));

    /* Set new vector's size */
    new_vector->size = size;


    cudaMalloc((void **) &(new_vector->data), size * sizeof(float));

    return new_vector;
}



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


    cudaMalloc((void **) &(new_matrix->data), num_rows * num_cols * sizeof(float));

    return new_matrix;
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

__global__
void calculate_matrix_times_vector_kernel(data_matrix *m,
                                          data_vector *v,
                                          data_vector *result) {



    /* TODO UNRESOLVED */




    int i, j;

    /* Put an assertion UNRESOLVED */

    for (i = 0; i < m->num_rows; i++) {

        result->data[i] = 0;

        for (j = 0; j < m->num_cols; j++) {
            result->data[i] += m->data[i * m->num_cols + j] * v->data[j];
        }
    }

}



/*
 * UNRESOLVED
 *
 */

__global__
void multiply_vector_by_constant_kernel(float *v, float c,
                                        float *result, int v_size) {

    int index;
    float temp;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < v_size;
         index += gridDim.x * blockDim.x) {

        temp = v[index] * c;

        result[index] = temp;

    }

}



void gpu_multiply_vector_by_constant(data_vector *v, float c,
                                     data_vector *result) {

    int threads_per_block;
    int blocks;

    threads_per_block = 512;

    blocks = v->size / threads_per_block;

    if (v->size % threads_per_block) {
        blocks++;
    }


    multiply_vector_by_constant_kernel<<<threads_per_block, blocks>>>
        (v->data, c, result->data, v->size);

}




__global__
void add_vectors_kernel(float *v1, float *v2, float *result, int v_size) {

    int index;
    float temp;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
         index < v_size;
         index += gridDim.x * blockDim.x) {

        temp = v1[index];

        temp += v2[index];

        result[index] = temp;

    }


}



/*
 * UNRESOLVED
 *
 */

void gpu_add_vectors(data_vector *v1, data_vector *v2, data_vector *result) {

    int threads_per_block;
    int blocks;

    threads_per_block = 512;

    blocks = v1->size / threads_per_block;

    if (v1->size % threads_per_block) {
        blocks++;
    }

    add_vectors_kernel<<<threads_per_block, blocks>>>
        (v1->data, v2->data, result->data, v1->size);

}


/*
 * UNRESOLVED
 *
 */

__global__
void compute_additive_inverse_of_vector_kernel(float *v, float *result, int v_size) {

    int index;
    float temp;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
        index < v_size;
        index += gridDim.x * blockDim.x) {

        temp = v[index];

        temp = -temp;

        result[index] = temp;

    }

}


void gpu_compute_additive_inverse_of_vector(data_vector *v, data_vector *result) {

    int threads_per_block;
    int blocks;

    threads_per_block = 512;

    blocks = v->size / threads_per_block;

    if (v->size % threads_per_block) {
        blocks++;
    }


    compute_additive_inverse_of_vector_kernel<<<threads_per_block, blocks>>>
        (v->data, result->data, v->size);

}



__global__
void add_constant_componentwise_to_vector_kernel(float *v, float c,
                                                 float *result, int v_size) {

    int index;
    float temp;

    for (index = blockDim.x * blockIdx.x + threadIdx.x;
        index < v_size;
        index += gridDim.x * blockDim.x) {

        temp = v[index];

        temp += c;

        result[index] = temp;

    }   

}




/*
 * UNRESOLVED
 *
 */

void gpu_add_constant_componentwise_to_vector(data_vector *v, float c,
                                              data_vector *result) {

    int threads_per_block;
    int blocks;

    threads_per_block = 512;

    blocks = v->size / threads_per_block;

    if (v->size % threads_per_block) {
        blocks++;
    }

    add_constant_componentwise_to_vector_kernel<<<threads_per_block, blocks>>>
        (v->data, c, result->data, v->size);   

}



/*
 * UNRESOLVED
 *
 */

void multiply_vectors_componentwise(data_vector *v1, data_vector *v2,
                                    data_vector *result) {

    int i;

    for (i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] * v2->data[i];
    }

}



/*
 * UNRESOLVED
 *
 */

void compute_matrix_transpose(data_matrix *m, data_matrix *result) {

    int i, j;

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

void apply_filter_to_vector_componentwise(data_vector *v,
                                          float (*filter)(float),
                                          data_vector *result) {

    int i;

    for (i = 0; i < v->size; i++) {
        result->data[i] = filter(v->data[i]);
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

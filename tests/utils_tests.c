#include "../src/utils/utils.h"
#include <stdio.h>
#include <stdlib.h>



int test1();
int test2();
int test3();
int test4();



int main() {

    printf("%d\n", test1());
    printf("%d\n", test2());
    printf("%d\n", test3());
    printf("%d\n", test4());

    return 0;
}


int test1() {

    float blah;

    data_vector *v;

    v = new_vector(5);

    free(v);

    return 1;
}


int test2() {

    data_vector *v1;
    data_vector *v2;
    float *arr1;
    float *arr2;
    float result;

    v1 = new_vector(5);
    v2 = new_vector(5);

    arr1 = (float *) (v1 + 1);
    arr2 = (float *) (v2 + 1);

    arr1[0] = 1;
    arr1[1] = 2;
    arr1[2] = 3;
    arr1[3] = 4;
    arr1[4] = 5;

    arr2[0] = -3;
    arr2[1] = 7;
    arr2[2] = 9;
    arr2[3] = 20;
    arr2[4] = -17;

    result = dot_product(v1, v2);

    return result == 33;
}



int test3() {

    data_matrix *m;
    data_vector *row;
    float *row_data;
    data_vector *input_vector;
    float *input_vector_data;
    data_vector *result;
    float *result_data;
    data_vector *expected_result;
    float *expected_result_data;
    int i;

    /* Create and populate matrix */
    m = new_data_matrix(3, 3);

    row = (data_vector *) (m + 1);
    row_data = (float *) (row + 1);

    row_data[0] = 1;
    row_data[1] = 7;
    row_data[2] = 2;

    row = (data_vector *) (((float *) (row + 1)) + row->size);
    row_data = (float *) (row + 1);

    row_data[0] = -2;
    row_data[1] = 9;
    row_data[2] = -3;

    row = (data_vector *) (((float *) (row + 1)) + row->size);
    row_data = (float *) (row + 1);

    row_data[0] = 4;
    row_data[1] = 5;
    row_data[2] = 6;


    /* Create and populate input vector */
    input_vector = new_vector(3);

    input_vector_data = (float *) (input_vector + 1);

    input_vector_data[0] = 1;
    input_vector_data[1] = -1;
    input_vector_data[2] = 5;


    /* Allocate space for result */
    result = new_vector(3);

    result_data = (float *) (result + 1);

    /* Create and populate expected result */
    expected_result = new_vector(3);

    expected_result_data = (float *) (expected_result + 1);

    expected_result_data[0] = 4;
    expected_result_data[1] = -26;
    expected_result_data[2] = 29;


    /* Compute matrix times vector */
    calculate_matrix_times_vector(m, input_vector, result);

    /* Compare result to expected result */
    for (i = 0; i < result->size; i++) {
        if (result_data[i] != expected_result_data[i]) {
            return 0;
        }
    }

    return 1;

}



float square_filter(float x) {
    return (x * x);
}

int test4() {

    data_vector *input_vector;
    float *input_vector_data;
    data_vector *result;
    float *result_data;
    data_vector *expected_result;
    float *expected_result_data;
    int i;


    /* Create and populate input vector */
    input_vector = new_vector(3);

    input_vector_data = (float *) (input_vector + 1);

    input_vector_data[0] = 7;
    input_vector_data[1] = 9;
    input_vector_data[2] = -2;

    /* Allocate space for result */
    result = new_vector(3);

    result_data = (float *) (result + 1);

    /* Create and populate expected result */
    expected_result = new_vector(3);

    expected_result_data = (float *) (expected_result + 1);

    expected_result_data[0] = 49;
    expected_result_data[1] = 81;
    expected_result_data[2] = 4;


    /* Run the filter */
    filter_vector(input_vector, result, &square_filter);


    /* Compare result to expected result */
    for (i = 0; i < result->size; i++) {
        if (result_data[i] != expected_result_data[i]) {
           return 0;
        }
    }

    return 1;

}

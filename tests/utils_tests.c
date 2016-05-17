#include "../src/utils/utils.h"
#include <stdio.h>
#include <stdlib.h>

int main() {

    printf("%d\n", test1());
    printf("%d\n", test2());

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



/*
data_matrix *new_data_matrix(int num_rows, int num_columns);
void calculate_matrix_times_vector(data_matrix *,
                                   data_vector *,
                                   data_vector *result);
void filter_vector(data_vector *input,
                   data_vector *output,
                   float (*filter)(float));

*/

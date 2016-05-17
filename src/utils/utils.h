

typedef struct {

    int size;

    /* Implicit array of floats */

} data_vector;


typedef struct {

    int num_rows;
    int num_cols;

    /* Implicit array of data_vectors */

} data_matrix;


data_vector *new_vector(int size);
float dot_product(data_vector *, data_vector *);
data_matrix *new_data_matrix(int num_rows, int num_columns);
void calculate_matrix_times_vector(data_matrix *,
                                   data_vector *,
                                   data_vector *result);
void filter_vector(data_vector *input,
                   data_vector *output,
                   float (*filter)(float));

/* Lots of methods to add UNRESOLVED */

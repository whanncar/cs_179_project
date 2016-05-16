

typedef struct {

    int size;

    /* Implicit array of floats */

} data_vector;


typedef struct {

    int num_rows;
    int num_cols;

    /* Implicit array of data_vectors */

} data_matrix;


data_vector *new_vector(int);
float *get_vector_data(data_vector *);
float dot_product(data_vector *, data_vector *);


/* Lots of methods to add UNRESOLVED */

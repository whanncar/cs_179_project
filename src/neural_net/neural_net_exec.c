#include "neural_net.h"


void set_neural_net_input(neural_net *nn, data_vector *input) {

    int i;
    float *nn_input_vector_data;
    float *input_vector_data;

    assert(input->size == nn->input_data->size);

    nn_input_vector_data = (float *) ((nn->input_data) + 1);
    input_vector_data = (float *) (input + 1);

    /* Copy input data into neural net */
    for (i = 0; i < input->size; i++) {
        nn_input_vector_data[i] = input_vector_data[i];
    }
}



void train_neural_net_on_input(neural_net *nn, data_vector *input) {

    /* Set neural net input */
    set_neural_net_input(nn, input);

    /* Forward propagate neural net */
    forward_propagate_neural_net(nn, NULL);

    /* Backward propagate neural net todo UNRESOLVED */

}



void train_neural_net(neural_net *nn) { /* This should be changed to allow file reading UNRESOLVED */

    data_vector *current_input;

    /* For each input vector */

        /* Train the neural net on the input */
        train_neural_net_on_input(nn, current_input);

}

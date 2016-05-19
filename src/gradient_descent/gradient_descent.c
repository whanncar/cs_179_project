


/* VERY MUCH IN NEED OF COMMENTING UNRESOLVED */

float partial_derivative(float calculated, float desired) {

    return 2 * (desired - calculated) * calculated * (1 - calculated);
}



void calculate_gradient(data_vector *input, float calculated,
                        float desired, data_vector *grad) {

    float *input_data;
    float *grad_data;
    float pd;
    int i;

    /* Get data arrays from vectors */
    input_data = (float *) (input + 1);
    grad_data = (float *) (input + 1);

    /* Calculate the "partial derivative" */
    pd = partial_derivative(calculated, desired);

    /* Calculate and store gradient */
    for (i = 0; i < input->size; i++) {
        grad_data[i] = input_data[i] * pd;
    }

}


void update_layer_weights(neural_layer *layer,
                          data_vector *desired,
                          float step_size) {

    data_vector *grad;
    float *output_data;
    float *desired_data;
    int i;

    output_data = (float *) ((layer->output_data) + 1);
    desired_data = (float *) (desired + 1);

    for (i = 0; i < layer->weights->num_rows; i++) {

        input = get_row(layer->weights, i);

        calculate_gradient(input, output_data[i], desired_data[i], grad);

    }

    /* finish me UNRESOLVED */

}

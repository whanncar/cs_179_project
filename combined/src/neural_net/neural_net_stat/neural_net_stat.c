#include "neural_net_stat.h"





float calculate_loss(neural_net *nn, sample_set *set) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    return calculate_matrix_distance(nn->output, set->sample_labels);

}



void predict(neural_net *nn, sample_set *set, data_matrix *predictions) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    int i, j;

    int rows, cols;

    data_matrix *output = nn->output;

    rows = output->num_rows;
    cols = output->num_cols;

    for (j = 0; j < cols; j++) {

        predictions->data[j] = 0;
        predictions->data[j + cols] = 0;

    }

    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            if (output->data[i * cols + j] > predictions->data[j]) {

                predictions->data[j] = output->data[i * cols + j];
                predictions->data[cols + j] = i;

            }

        }

    }

}


float calculate_percent_predicted_correctly(neural_net *nn, sample_set *set) {

    int i;
    float count;

    data_matrix *predictions;

    predictions = new_matrix(2, set->sample_inputs->num_cols);

    predict(nn, set, predictions);

    count = 0;

    for (i = 0; i < set->sample_inputs->num_cols; i++) {

        if (set->sample_labels->data[((int) (predictions->data[predictions->num_cols + i])) * set->sample_inputs->num_cols + i]) {

            count++;

        }

    }

    free_matrix(predictions);

    return 100 * count / ((float) set->sample_inputs->num_cols);

}

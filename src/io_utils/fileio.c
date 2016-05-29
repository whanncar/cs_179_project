#include "fileio.h"

#define NUM_DIGITS 10



/*
 * UNRESOLVED
 *
 */

int *get_data(char *line, int sample_length) {

    int *data;
    int i, j;
    char *temp;

    data = (int *) malloc(sample_length * sizeof(int)); 

    i = 0;

    for (j = 0; line[j] != '\0'; j++) {

        if (line[j] == ',') {
            data[i] = j;
            i++;
        }

    }

    for (i = 0; i < sample_length - 1; i++) {

        temp = (char *) malloc((data[i + 1] - data[i]) * sizeof(char));

        for (j = 0; j < data[i + 1] - data[i] - 1; j++) {
            temp[j] = line[j + data[i] + 1];
        }

        temp[j] = '\0';

        data[i] = atoi(temp);

    }

    for (j = data[sample_length - 1]; line[j] != '\0'; j++);

    temp = (char *) malloc((j - data[sample_length - 1]) * sizeof(char));

    for (; j > data[sample_length - 1]; j--) {

        temp[j - data[sample_length - 1] - 1] = line[j];

    }

    data[sample_length - 1] = atoi(temp);

    return data;

}



/*
 * UNRESOLVED
 *
 */

int get_label(char *line) {

    return (int) (line[0] - '0');

}


/*
 * UNRESOLVED
 *
 */

sample_set *get_samples_from_file(char *filepath,
                                  int num_samples,
                                  int sample_length) {

    char *line;
    FILE *fstream;
    sample_set *samples;
    int i, j;
    int label;
    int *data;

    data_matrix *sample_inputs_T;
    data_matrix *sample_labels_T;


    samples = (sample_set *) malloc(sizeof(sample_set));

    samples->sample_inputs = new_matrix(sample_length, num_samples);
    sample_inputs_T = new_matrix(num_samples, sample_length);

    samples->sample_labels = new_matrix(NUM_DIGITS, num_samples);
    sample_labels_T = new_matrix(num_samples, NUM_DIGITS);

    fstream = fopen(filepath, "r");

    line = (char *) malloc((sample_length + 1) * 4 * sizeof(char));

    i = 0;

    while (fgets(line, (sample_length + 1) * 4, fstream) != NULL
           && i < num_samples) {

        label = get_label(line);
        data = get_data(line, sample_length);

        for (j = 0; j < NUM_DIGITS; j++) {
            if (j == label) {
                sample_labels_T->data[i * NUM_DIGITS + j] = 1;
            }
            else {
                sample_labels_T->data[i * NUM_DIGITS + j] = 0;
            }
        }

        for (j = 0; j < sample_length; j++) {
            sample_inputs_T->data[i * sample_length + j] = (float) data[j];
        }

        free(data);

        i++;
    }

    fclose(fstream);

    compute_matrix_transpose(sample_labels_T, samples->sample_labels);
    compute_matrix_transpose(sample_inputs_T, samples->sample_inputs);

    free_matrix(sample_labels_T);
    free_matrix(sample_inputs_T);

    return samples;

}

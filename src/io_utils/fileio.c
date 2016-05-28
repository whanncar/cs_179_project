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


    samples = (sample_set *) malloc(sizeof(sample_set));

    samples->num_samples = num_samples;

    samples->sample_ptrs = (sample **) malloc(num_samples * sizeof(sample *));

    for (i = 0; i < num_samples; i++) {

        samples->sample_ptrs[i] = (sample *) malloc(sizeof(sample));
        samples->sample_ptrs[i]->input = new_vector(sample_length);
        samples->sample_ptrs[i]->expected_output = new_vector(NUM_DIGITS);
    }

    fstream = fopen(filepath, "r");

    line = (char *) malloc((sample_length + 1) * 4 * sizeof(char));

    i = 0;

    while (fgets(line, (sample_length + 1) * 4, fstream) != NULL
           && i < num_samples) {

        label = get_label(line);
        data = get_data(line, sample_length);

        for (j = 0; j < NUM_DIGITS; j++) {
            if (j == label) {
                samples->sample_ptrs[i]->expected_output->data[j] = 1;
            }
            else {
                samples->sample_ptrs[i]->expected_output->data[j] = 0;
            }
        }

        for (j = 0; j < samples->sample_ptrs[i]->input->size; j++) {
            samples->sample_ptrs[i]->input->data[j] = (float) data[j];
        }

        free(data);

        i++;
    }

    fclose(fstream);

    return samples;

}

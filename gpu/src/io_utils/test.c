#include "fileio.h"


int main() {

    sample_set *set;
    sample *current_sample;
    int i, j;

    char *filepath = "../../mnist_test.csv";

    set = get_samples_from_file(filepath, 2, 784); 

    for (i = 0; i < set->num_samples; i++) {

        current_sample = set->sample_ptrs[i];

        for (j = 0; j < current_sample->expected_output->size; j++) {

            printf("%f\n", current_sample->expected_output->data[j]);

        }

        printf("\n");

        for (j = 0; j < current_sample->input->size; j++) {

            printf("%f\n", current_sample->input->data[j]);

        }

        printf("\n");

    }


    return 0;

}

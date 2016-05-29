#include <stdio.h>
#include <stdlib.h>
#include "./neural_net/neural_net.h" 
#include "io_utils/fileio.h"


neural_net *nn;
sample_set *samples;

void build_neural_net_from_cmds(int, char **);


int main(int argc, char **argv) {

    float old_loss;
    float new_loss;
    float epsilon;
    float lambda;

    epsilon = .1;
    lambda = .001;

    build_neural_net_from_cmds(argc, argv);

    samples = get_samples_from_file("training_data/mnist_test.csv", 1000, 784);   

    old_loss = calculate_loss(nn, samples);

printf("%f\n", old_loss);

    new_loss = old_loss + 1;

    while ((old_loss - new_loss) * (old_loss - new_loss) > epsilon) {

        old_loss = new_loss;

        train_neural_net(nn, samples, lambda);

        new_loss = calculate_loss(nn, samples); 

        printf("%f, %f\n", new_loss, calculate_percent_predicted_correctly(nn, samples));

    }
 

}



/*
 * UNRESOLVED
 *
 */

void build_neural_net_from_cmds(int argc, char **argv) {

    int i, j;
    int num_layers;
    int *layer_specs;

    num_layers = atoi(argv[1]);

    layer_specs = (int *) malloc((num_layers + 1) * sizeof(int));

    for (i = 0; i < num_layers + 1; i++) {
        layer_specs[i] = atoi(argv[i + 2]);
    }

    nn = (neural_net *) malloc(sizeof(neural_net));

    nn->num_layers = num_layers;

    nn->layer_ptrs = (neural_layer **) malloc(num_layers * sizeof(neural_layer *));

    for (i = 0; i < num_layers; i++) {

        nn->layer_ptrs[i] = (neural_layer *) malloc(sizeof(neural_layer));

    }

    nn->layer_ptrs[0]->input = new_vector(layer_specs[0]);

    for (i = 0; i < num_layers - 1; i++) {

        nn->layer_ptrs[i]->output = new_vector(layer_specs[i + 1]);

        nn->layer_ptrs[i + 1]->input = nn->layer_ptrs[i]->output;

    }

    nn->layer_ptrs[num_layers - 1]->output = new_vector(layer_specs[num_layers]);

    nn->input = nn->layer_ptrs[0]->input;
    nn->output = nn->layer_ptrs[num_layers - 1]->output;

    for (i = 0; i < num_layers; i++) {

        nn->layer_ptrs[i]->r = new_vector(nn->layer_ptrs[i]->output->size);
        nn->layer_ptrs[i]->t = new_vector(nn->layer_ptrs[i]->output->size);
        nn->layer_ptrs[i]->s = new_vector(nn->layer_ptrs[i]->output->size);
        nn->layer_ptrs[i]->dL_ds_local = new_vector(nn->layer_ptrs[i]->output->size);
        nn->layer_ptrs[i]->dL_ds_global = new_vector(nn->layer_ptrs[i]->output->size);

        for (j = 0; j < nn->layer_ptrs[i]->t->size; j++) {
            nn->layer_ptrs[i]->t->data[j] = 1;
        }

    }

    for (i = 0; i < num_layers; i++) {

        nn->layer_ptrs[i]->w = new_matrix(nn->layer_ptrs[i]->output->size,
                                          nn->layer_ptrs[i]->input->size);

        fill_matrix_rand(nn->layer_ptrs[i]->w, -.5, .5);

        nn->layer_ptrs[i]->w_T = new_matrix(nn->layer_ptrs[i]->w->num_cols,
                                            nn->layer_ptrs[i]->w->num_rows);

        compute_matrix_transpose(nn->layer_ptrs[i]->w, nn->layer_ptrs[i]->w_T);
    }

}

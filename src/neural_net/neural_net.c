#include "../utils/utils.h"
#include "neural_net.h"
#include <math.h>



/*
 * sigmoid_filter: Computes the sigmoid function
 *
 * x: Input value to sigmoid
 *
 * return value: Output value from sigmoid
 *
 * where should I live? UNRESOLVED
 *
 */

float sigmoid_filter(float x) {

    return 1 / (1 + expf(-x));

}




/*
 * forward_propagate_layer: Forward propagates the given neural layer
 *                          using the given filter
 *
 * arguments: layer: The neural layer to forward propagate
 *            filter: The filter applied to the weighted sums
 *
 */

void forward_propagate_layer(neural_layer *layer, float (*filter)(float)) {

    int i;
    float *out_data;

    /* Calculate the weighted sums */
    calculate_matrix_times_vector(layer->weights,
                                  layer->input_data,
                                  layer->output_data);

    /* Get the output data array */
    out_data = (float *) (layer->output_data + 1);

    /* Apply the sigmoid filter to the output data */
    for (i = 0; i < layer->output_data->size; i++) {
        out_data[i] = sigmoid_filter(out_data[i]);
    }

}



/*
 * forward_propagate_neural_net: Forward propagates the given neural net
 *                               using the given filter
 *
 * arguments: nn: The neural net to be forward propagated
 *            filter: The filter applied to the weighted sums at each layer
 *
 */

void forward_propagate_neural_net(neural_net *nn) {

    int i;
    int num_layers;
    neural_layer *layers;

    /* Store neural net structural data locally */
    num_layers = nn->num_layers;
    layers = *(nn->layers_ptrs);

    /* Propagate each layer */
    for (i = 0; i < num_layers; i++) {
        forward_propagate_layer(layers + i, filter);
    }

}



/*
 *
 *
 */

void backward_propagate_neural_layer(neural_layer *layer) {

    

}




/*
 * free_neural_net: Frees the given neural net and
 *                  all of its associated data
 *
 * arguments: nn: Neural net to be freed
 *
 */

void free_neural_net(neural_net *nn) {

    neural_layer *layer;
    int i;

    /* For each neural net layer */
    for (i = 0; i < nn->num_layers; i++) {

        layer = (nn->layer_ptrs)[i];

        /* Free layer's input data and weights */
        free(layer->input_data);
        free(layer->weights);
    }

    /* Free output data */
    free(nn->output_data);

    /* Free array of pointers to layers */
    free(nn->layer_ptrs);

    free(nn);
}

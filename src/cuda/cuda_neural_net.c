//GPU implementatiion

gpu_initialize_neural_net(argc, argv);


void gpu_initialize_neural_net(int argc, char **argv) {

    int num_layers;
    int num_inputs;
    int input_size;
    int output_size;
    int *layer_weight_specs;
    int i;   

    num_inputs = atoi(argv[1]);
    input_size = atoi(argv[2]);
    output_size = atoi(argv[3]);
    num_layers = atoi(argv[4]);

    layer_weight_specs = (int *) malloc(num_layers * sizeof(int));

    for (i = 0; i < num_layers; i++) {
        layer_weight_specs[i] = atoi(argv[5 + i]);
    }

    num_layers += 1;

    nn = gpu_new_neural_net(num_layers, num_inputs, input_size, output_size, layer_weight_specs);

    initialize_neural_net_weights(nn);

    free(layer_weight_specs);
}


neural_net *gpu_new_neural_net(int num_layers, int num_inputs,
                           int input_size, int output_size,
                           int *layer_weight_specs)
{
	int i;

    neural_net *gpu_nn;

    cudaMalloc((void *)&gpu_nn, sizeof(neural_net));

    cudaMemcpy(gpu_nn->num_layers, num_layers, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&gpu_nn->layer_ptrs, num_layers * sizeof(neural_layer *));

    for (i = 1; i < num_layers - 1; i++) {
        gpu_nn->layer_ptrs[i] = gpu_new_neural_layer(layer_weight_specs[i - 1],
                                             layer_weight_specs[i],
                                             num_inputs);
    }

    gpu_nn->layer_ptrs[i] = gpu_new_neural_layer(layer_weight_specs[i - 1],
                                         output_size,
                                         num_inputs);


    /* Connect inputs and outputs of adjacent layers */
    nn->layer_ptrs[0]->input = gpu_new_matrix(input_size, num_inputs);

    for (i = 1; i < num_layers; i++) {
        gpu_nn->layer_ptrs[i]->input = gpu_nn->layer_ptrs[i - 1]->output;
    }


    gpu_nn->input = gpu_nn->layer_ptrs[0]->input;
    gpu_nn->output = gpu_nn->layer_ptrs[num_layers - 1]->output;

    return gpu_nn;
}


neural_layer *gpu_new_neural_layer(int input_length,
                               int num_weights,
                               int num_inputs) {

    neural_layer *layer;

    cudaMalloc((void *)&layer, sizeof(neural_layer));
    
    layer->w = gpu_new_matrix(num_weights, input_length);

    layer->w_T = gpu_new_matrix(input_length, num_weights);

    layer->s = gpu_new_matrix(num_weights, num_inputs);

    layer->output = gpu_new_matrix(num_weights, num_inputs);

    layer->dL_ds = gpu_new_matrix(num_weights, num_inputs);

    layer->dL_dw = gpu_new_matrix(num_weights, input_length);

    return layer;

}


data_matrix *gpu_new_matrix(int num_rows, int num_cols) {

    data_matrix *new_matrix;

    /* Allocate space for new matrix */

	cudaMalloc((void *)&new_matrix, sizeof(data_matrix));

	/* Initialize new matrix's structure values */
	cudaMemcpy(new_matrix->num_rows, num_rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(new_matrix->num_cols, num_cols, sizeof(int), cudaMemcpyHostToDevice);

    /* Allocate space for new matrix's data */
    cudaMalloc((void *)&new_matrix->data, num_rows * num_cols * sizeof(float));
   
    /* Set the matrix's stride */
    cudaMemcpy(new_matrix->stride, num_cols, sizeof(int), cudaMemcpyHostToDevice);
    
    return new_matrix;
}


void gpu_initialize_neural_net_weights(neural_net *nn) {

    int i;
    neural_layer *layer;

    for (i = 0; i < nn->num_layers; i++) {

        //layer = nn->layer_ptrs[i];
        cudaMemcpy(layer , nn->layer_ptrs[i], sizeof(neural_layer *), cudaMemcpyDeviceToHost);

        fill_matrix_rand(layer->w, -.5, .5);

        compute_matrix_transpose(layer->w, layer->w_T);

        cudaMemcpy(nn->layer_ptrs[i]->w_T, layer->w_T, sizeof(new_matrix *), cudaMemcpyHostToDevice);

    }

}
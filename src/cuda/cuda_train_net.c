///GPU initialize samples

//done once on CPU
initialize_samples(argc, argv);

train_neural_net(nn, samples, lambda);

//Nothing changed in this function between cpu and GPU implmentation
void train_neural_net(neural_net *nn, sample_set *set, float step) {

    set_neural_net_input(nn, set->sample_inputs);

    forward_propagate_neural_net(nn);

    backward_propagate_neural_net(nn, set->sample_labels, step);

}

cudaMemcpy(gpu_nn->num_layers, num_layers, sizeof(int), cudaMemcpyHostToDevice);
cudaMalloc((void **)&gpu_nn->layer_ptrs, num_layers * sizeof(neural_layer *));

void gpu_set_neural_net_input(neural_net *nn, data_matrix *input) {


    int i, j;

    int rows, cols;

    rows = nn->input->num_rows;
    cols = nn->input->num_cols;

    cudaMemcpy(rows, nn->input->num_rows, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cols, nn->input->num_cols, sizeof(int), cudaMemcpyDeviceToHost);

//Not the most efficient but should work.  This only needs to be called once
    for (i = 0; i < rows; i++) {

        for (j = 0; j < cols; j++) {

            //nn->input->data[i * cols + j] = input->data[i * cols + j];
            cudaMemcpy(nn->input->data[i * cols + j], input->data[i * cols + j], sizeof(float), cudaMemcpyHostToDevice);

        }

    }

}

//not sure if nn->num_layers can be accese
void forward_propagate_neural_net(neural_net *nn) {

    int i;
    //int numLayers;

    //cudaMemcpy(numLayers, nn->num_layer, sizeof(int), cudaMemcpyDeviceToHost);

    //Layers are processed one at a time by returning to cpu before calling
    //the next forward propagate.  This is required since the second layer
    //can only be computed after the first one is finished.  
    /* Propagate each layer */
    for (i = 0; i < nn->num_layer; i++) {
        forward_propagate_layer(nn->layer_ptrs[i]);
    }

}
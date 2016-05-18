


neural_net *nn;
data_vector *input;
data_vector *output;

int main() {

    /* Get config from config file */

    /* Make neural net from config */
    nn = new_neural_net(config);

    /* For each training vector */

        /* Get input vector */

        /* Train on input vector */
        train_neural_net_on_input(input);    

}




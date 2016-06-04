Final project readme
CS179 
Wade Hann-Caruthers and Andrew Janzen
California Institute of Technology
6-3-2016

//Installation and Usage instructions
Navigate to the src folder via the 
	cd folder 	
compile the code by calling     
	make
//The make file was constructed for UNIX AND WILL NOT WORK ON WINDOWS

Once compiled, call the .exe file to run the neural network
	run_neural_net.exe  <number of samples>  <sample length> <label length> <number of hidden layers>  <layer 1 size> ... <layer n size> 
Where 	number of samples is the number of input images used to train the network
sample length is the unwrapped length of each image (IMAGE_DIM*IMAGE_DIM)
label length is the number of output labels the network is design to clasify
number of hidden layers  is the number of neural network layers - 1
layer X size is the number of neurons to be used on layer
The network takes as its input a training file containing 10000 images of hand written digits of size 28*28 with labels associating each image with a number (0 - 9).  
An acceptable executable call would be
	run_neural_net.exe  10000   784  10  2  784  90 
which would build and train a network on all 10000 samples, each of size 784, with 10 output number neurons, 2 hidden layers, 784 first layer neurons, and 90 second layer neurons.  

WHAT THE PROGRAM DOES
The program is designed to train a convolutional neural net to recoginize hand written digits from images.  A training set (mnist_test.csv) is used as the sample dataset. 
The program load the data from the file, generates a neural net struct and initializes all the weights to random values.  
The program iterates on the input data, running it through the network, checking the estimated outputs against "ground truth" and then back propagating the error through the network.  After reaching back to the input layer, the weights for each layer are updated to move each weight closer to its optimal.  The network runs for a specified number of interations define in the code.  
The GPU is used to accelerate forward and back propagation as these are by far the most computationally intensive and most repeated parts of the code.
The network implements a single convolutional layer at the input layer.  This allows for slightly more accurate image recognition to shifts in the positions of the numbers in the images.  

EXPECTED RESULTS
The network is expected to quickly converge on an optimal set of weights.  Although it might appear that the implementation takes a while, a CPU implementation is still much slower.   The criterion for optimality is the "Loss" of the network which is basically the sum of the squared error for the output neurons.  

ANALYSIS of PERFORMANCE
10000 samples, 100 iterations, 3 layers [784, 90,10]
102 seconds to go through 10 iterations on CPU
102 seconds to go through 600 iterations on GPU
This means that using the GPU gives a 60X speedup.  

for a network of 1000 samples, 3 layers [784, 90 10]
GPU time 25 seconds
Matlab time 2.02 minutes
Note that Matlab used extremely optimized matrix multiplies to implement the entire network.  
4x speedup comparision with Matlab



	


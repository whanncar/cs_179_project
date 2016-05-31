

all: run_neural_net

run_neural_net:
	nvcc ../src/utils/utils.c src/utils/utils.c src/neural_net/neural_layer.c src/neural_net/neural_net.c src/main.c -o gpu_run_neural_net

clean:
	rm gpu_run_neural_net

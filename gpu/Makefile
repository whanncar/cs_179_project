

all: run_neural_net

run_neural_net:
	nvcc ../src/utils/utils.c src/utils/utils.cu src/neural_net/neural_layer.cu src/neural_net/neural_net.cu src/main.cu -o gpu_run_neural_net

clean:
	rm gpu_run_neural_net

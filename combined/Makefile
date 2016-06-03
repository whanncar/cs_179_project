

all: run_neural_net

run_neural_net:
	nvcc src/utils/utils.c src/utils/gpu_utils.c src/utils/gpu_utils.cu src/neural_net/neural_layer.c src/neural_net/gpu_neural_layer.c src/neural_net/neural_net.c src/neural_net/gpu_neural_net.c src/neural_net/neural_net_exec/gpu_neural_net_exec.c src/neural_net/neural_net_exec/gpu_neural_net_exec_utils.c src/neural_net/neural_net_exec/neural_net_exec.c src/neural_net/neural_net_exec/neural_net_exec_utils.c src/neural_net/neural_net_stat/gpu_neural_net_stat.c src/neural_net/neural_net_stat/neural_net_stat.c io_utils/fileio.c src/main.c -o gpu_run_neural_net

clean:
	rm gpu_run_neural_net

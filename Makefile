

all: run_neural_net

run_neural_net:
	gcc src/main.c src/utils/utils.c src/neural_net/neural_net.c src/io_utils/fileio.c -lm -o run_neural_net

clean:
	rm run_neural_net

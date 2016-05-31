

all: run_neural_net

run_neural_net:
	gcc src/utils/utils.c src/neural_net/neural_layer.c src/neural_net/neural_net.c src/neural_net/neural_net_exec/neural_net_exec_utils.c src/neural_net/neural_net_exec/neural_net_exec.c src/neural_net/neural_net_stat/neural_net_stat.c src/io_utils/fileio.c src/main.c -lm -o run_neural_net

run_neural_net_bad: utils.o neural_layer.o neural_net.o neural_net_exec.o neural_net_stat.o fileio.o
	gcc src/main.c -c
	gcc neural_layer.o neural_net.o neural_net_exec.o neural_net_stat.o fileio.o main.o -o run_neural_net

utils.o: src/utils/utils.h
	gcc src/utils/utils.c -c
	cp src/utils/utils.h .

neural_layer.o: src/neural_net/neural_layer.h utils.o
	gcc src/neural_net/neural_layer.c -c
	cp src/neural_net/neural_layer.h .

neural_net.o: src/neural_net/neural_net.h neural_layer.o
	gcc src/neural_net/neural_net.c -c
	cp src/neural_net/neural_net.h .

neural_net_exec.o: src/neural_net/neural_net_exec/neural_net_exec.h src/neural_net/neural_net_exec/neural_net_exec_utils.h neural_net.o
	gcc src/neural_net/neural_net_exec/neural_net_exec.c src/neural_net/neural_net_exec/neural_net_exec_utils.c -lm -c
	cp src/neural_net/neural_net_exec/neural_net_exec.h .
	cp src/neural_net/neural_net_exec/neural_net_exec_utils.h .

neural_net_stat.o: src/neural_net/neural_net_stat/neural_net_stat.h neural_net_exec.o
	gcc src/neural_net/neural_net_stat/neural_net_stat.c -c
	cp src/neural_net/neural_net_stat/neural_net_stat.h .

fileio.o: src/io_utils/fileio.h neural_net.o
	gcc src/io_utils/fileio.c -c
	cp src/io_utils/fileio.h .

tidy:
	mv *.o *.h src/objs

clean:
	rm *.o src/objs/*.o run_neural_net

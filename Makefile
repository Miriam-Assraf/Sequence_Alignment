build:
	mpicxx -fopenmp -c Main.c -o Main.o
	mpicxx -fopenmp -c Functions.c -o Functions.o
	mpicxx -fopenmp -c Sequential.c -o Sequential.o
	nvcc -I./inc -c CudaFunctions.cu -o CudaFunctions.o
	mpicxx -fopenmp -o SequenceAlignment Main.o Functions.o Sequential.o CudaFunctions.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./SequenceAlignment

run:
	mpiexec -np 6 ./SequenceAlignment

runOn2:
	mpiexec -np 3 --machinefile hosts.txt --map-by node ./SequenceAlignment
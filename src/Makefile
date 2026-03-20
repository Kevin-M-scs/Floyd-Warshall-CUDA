#-*-Makefile-*-

main: main.o Graph.o floyd_warshall.o
	nvcc -m64 -O2 main.o Graph.o floyd_warshall.o -o main

main.o: main.cpp Graph.h floyd_warshallCUDA.h
	nvcc -m64 -O2 -c main.cpp

Graph.o: Graph.cpp Graph.h
	nvcc -m64 -O2 -c Graph.cpp

floyd_warshall.o: floyd_warshall.cu floyd_warshallCUDA.h
	nvcc -m64 -O2 -c floyd_warshall.cu -o floyd_warshall.o

clean:
	rm main.o Graph.o floyd_warshall.o main.exe




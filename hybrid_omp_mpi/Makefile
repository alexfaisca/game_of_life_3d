MPICC = mpicc

COMPILER_FLAGS = -fopenmp -ansi -std=c11 -pedantic -ggdb -O2

SRCS := $(wildcard *.c)
BINS := life3d-hybrid

all: ${BINS}

life3d-hybrid: life3d-hybrid-mpi-omp.c
	@echo "Compiling..."
	${MPICC} ${COMPILER_FLAGS} $< -o $@

clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS}

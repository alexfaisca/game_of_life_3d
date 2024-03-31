MPICC = mpicc

COMPILER_FLAGS = -ansi -std=c11 -pedantic -fopenmp -ggdb -O2

SRCS := $(wildcard *.c)
BINS := life3d-mpi

all: ${BINS}

%: %.c
	@echo "Compiling..."
	${MPICC} ${COMPILER_FLAGS} $< -o $@

clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS}

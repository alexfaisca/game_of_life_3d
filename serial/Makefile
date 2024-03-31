CC = gcc

COMPILER_FLAGS = -ansi -std=c11 -pedantic -fopenmp -ggdb -O2

SRCS := $(wildcard *.c)
BINS := life3d

all: ${BINS}

%: %.c
	@echo "Compiling..."
	${CC} ${COMPILER_FLAGS} $< -o $@

clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS}

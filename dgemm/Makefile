# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icc
OPT = -O3
CFLAGS = -qopt-report -qopt-report-phase=vec -Wall -DGETTIMEOFDAY -std=c99 -fopenmp -xAVX2 $(OPT)
LDFLAGS = -Wall -parallel
# mkl is needed for blas implementation
LDLIBS = -parallel -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark-test.o benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-naive : benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)

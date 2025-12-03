FLAGS= -DDEBUG
LIBS= -lm
CUDA_LIBS= -lcuda -lcudart
ALWAYS_REBUILD=makefile
NVCC=nvcc
CC=gcc

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS) $(CUDA_LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
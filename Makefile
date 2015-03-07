CC=nvcc
CFLAGS=-I/usr/local/cuda-5.5/include -I/cfs/zorn/nobackup/m/mircom/cuda/samples/common/inc
LDFLAGS=-L${CUDA_HOME}/lib64 

all: cudca cudca_opt

cudca: cudca.cu
	$(CC) -o cudca $(CFLAGS) -arch sm_20 $(LDFLAGS) cudca.cu
	
cudca_opt: cudca_opt.cu
	$(CC) -o cudca_opt $(CFLAGS) -arch sm_20 $(LDFLAGS) cudca_opt.cu

clean: 
	rm cudca cudca_opt

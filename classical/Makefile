NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -use_fast_math

TARGET = labs_gpu
SRCS = labs_gpu.cu kernels.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

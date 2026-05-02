CUDA_PATH   ?= /usr/local/cuda-13.1
NVCC        := $(CUDA_PATH)/bin/nvcc

# RTX 4080 Laptop = sm_89; PTX fallback (compute_89) for JIT on newer GPUs
ARCH        := -gencode arch=compute_89,code=sm_89 \
               -gencode arch=compute_89,code=compute_89

NVCCFLAGS   := -O3 --compiler-options '-fPIC' $(ARCH)

INCLUDES    := -I$(CUDA_PATH)/targets/x86_64-linux/include
LDFLAGS     := -L$(CUDA_PATH)/targets/x86_64-linux/lib \
               -lnppig -lnppc -lcudart \
               -shared -Xlinker -soname=libmei_remap.so

TARGET      := libmei_remap.so
SRC         := mei_remap.cu

all: $(TARGET)

$(TARGET): $(SRC) mei_remap.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<
	@echo "Built: $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: all clean

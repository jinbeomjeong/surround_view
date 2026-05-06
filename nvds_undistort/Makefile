DS_ROOT     ?= /opt/nvidia/deepstream/deepstream-9.0
CUDA_PATH   ?= /usr/local/cuda-13.1
NVCC        := $(CUDA_PATH)/bin/nvcc

# RTX 4080 Laptop = sm_89
ARCH        := -gencode arch=compute_89,code=sm_89 \
               -gencode arch=compute_89,code=compute_89

NVCCFLAGS   := -O3 --compiler-options '-fPIC' $(ARCH)

# strip -pthread from GST_CFLAGS (nvcc doesn't accept it; handled by g++ host)
GST_CFLAGS  := $(filter-out -pthread, $(shell pkg-config --cflags gstreamer-1.0))

INCLUDES    := \
  -I$(CUDA_PATH)/targets/x86_64-linux/include \
  -I$(DS_ROOT)/sources/includes \
  -I$(DS_ROOT)/sources/gst-plugins/gst-nvdspreprocess/include \
  -I$(DS_ROOT)/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib \
  $(GST_CFLAGS)

LDFLAGS     := \
  -L$(CUDA_PATH)/targets/x86_64-linux/lib -lnppig -lnppc -lcudart \
  -L$(DS_ROOT)/lib -lnvbufsurface \
  -shared -Xlinker -soname=libmei_undistort.so

TARGET      := libmei_undistort.so
SRC         := remap_lib.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<
	@echo "Built: $(TARGET)"

maps: save_mei_undistort_maps.py
	cd .. && venv/bin/python nvds_undistort/save_mei_undistort_maps.py

clean:
	rm -f $(TARGET)

.PHONY: all maps clean

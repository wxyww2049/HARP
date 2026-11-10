PYTHON_BIN ?= python
RING_ATTN_CXX ?= mpicxx

KBLAS_INCLUDE ?= $(HPCKIT_ROOT)/kml/bisheng/include
KBLAS_LIBDIR ?= $(HPCKIT_ROOT)/kml/bisheng/lib/sme/kblas/multi
ARCH_FLAGS ?= -march=armv9-a+sme+bf16+sve

BUILD_DIR := build
TARGET := $(BUILD_DIR)/libring_attention_ops.1.so
KERNELS_DIR := ../../kernels
FINAL_TARGET := $(KERNELS_DIR)/libring_attention_ops.so
SRCS := torch_ring_attention_ops.cpp ring_attn.cpp

TORCH_INCLUDE_FLAGS := $(shell $(PYTHON_BIN) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + p for p in include_paths()))")
TORCH_LIB_FLAGS := $(shell $(PYTHON_BIN) -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L{0} -Wl,-rpath,{0}'.format(p) for p in library_paths()))")
TORCH_ABI := $(shell $(PYTHON_BIN) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

CXXFLAGS := -std=c++17 -O3 -g -shared -fPIC -fopenmp -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) $(ARCH_FLAGS)
CPPFLAGS := -I$(KBLAS_INCLUDE) $(TORCH_INCLUDE_FLAGS)
LDFLAGS := -L$(KBLAS_LIBDIR) -Wl,-rpath,$(KBLAS_LIBDIR) $(TORCH_LIB_FLAGS)
LDLIBS := -ltorch_cpu -ltorch -lc10 -lkblas -lnuma -lm --rtlib=compiler-rt

.PHONY: all clean print-vars

all: $(FINAL_TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(KERNELS_DIR):
	mkdir -p $@

$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(RING_ATTN_CXX) $(CXXFLAGS) $(SRCS) $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) -o $@

$(FINAL_TARGET): $(TARGET) | $(KERNELS_DIR)
	cp -f $< $@

clean:
	rm -f $(TARGET) $(FINAL_TARGET)

print-vars:
	@echo "PYTHON_BIN=$(PYTHON_BIN)"
	@echo "RING_ATTN_CXX=$(RING_ATTN_CXX)"
	@echo "TORCH_ABI=$(TORCH_ABI)"
	@echo "TORCH_INCLUDE_FLAGS=$(TORCH_INCLUDE_FLAGS)"
	@echo "TORCH_LIB_FLAGS=$(TORCH_LIB_FLAGS)"

ifeq ($(origin CXX),default)
CXX := clang++
endif
MPI_CXX ?= mpicxx

ARCH_FLAGS := -march=armv9-a+sme+bf16+sve
COMMON_FLAGS := -O3 -std=c++20 $(ARCH_FLAGS) -g -fopenmp --rtlib=compiler-rt
PIC_FLAGS := $(COMMON_FLAGS) -fPIC
LDFLAGS_SO := -shared -fopenmp -lnuma -lmemkind --rtlib=compiler-rt
LDFLAGS_EXE := -fopenmp -lnuma -lmemkind --rtlib=compiler-rt
MPI_CFLAGS := $(shell $(MPI_CXX) --showme:compile 2>/dev/null)
MPI_LDFLAGS := $(shell $(MPI_CXX) --showme:link 2>/dev/null)

KERNELS_DIR := ../../kernels
BUILD_DIR := build
LIB_NAME := libsmeblas.so
TEST_NAME := bgemm_test
FINAL_LIB := $(KERNELS_DIR)/libsmeblas.so

DISPATCH_SRC := bgemm.cpp
HEADER := bgemm.h ../memory_node.h

DEMO1_SRC := ../bgemm_demo1/bgemm.cpp
DEMO2_SRC := ../bgemm_demo2/gemm.cpp
OUTER_SRC := ../bgemm_outer/outer_gemm.cpp
VECTOR_SRC := ../bgemm_vector/gemm.cpp

DISPATCH_OBJ := $(BUILD_DIR)/bgemm_dispatch.o
TEST_MAIN_OBJ := $(BUILD_DIR)/bgemm_test_main.o
DEMO1_OBJ := $(BUILD_DIR)/bgemm_demo1_impl.o
DEMO2_OBJ := $(BUILD_DIR)/bgemm_demo2_impl.o
OUTER_OBJ := $(BUILD_DIR)/bgemm_outer_impl.o
VECTOR_OBJ := $(BUILD_DIR)/bgemm_vector_impl.o

BACKEND_OBJS := $(DEMO1_OBJ) $(DEMO2_OBJ) $(OUTER_OBJ) $(VECTOR_OBJ)

.PHONY: all lib test clean

all: lib test

lib: $(FINAL_LIB)

test: $(TEST_NAME)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(KERNELS_DIR):
	mkdir -p $(KERNELS_DIR)

$(DISPATCH_OBJ): $(DISPATCH_SRC) $(HEADER) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -c $< -o $@

$(TEST_MAIN_OBJ): $(DISPATCH_SRC) $(HEADER) | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(MPI_CFLAGS) -DFINAL_DEMO_TEST_MAIN -c $< -o $@

$(DEMO1_OBJ): $(DEMO1_SRC) shared_workspace.h ../memory_node.h | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=bgemm_demo1_impl -c $< -o $@

$(DEMO2_OBJ): $(DEMO2_SRC) shared_workspace.h ../memory_node.h | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=bgemm_demo2_impl -c $< -o $@

$(OUTER_OBJ): $(OUTER_SRC) shared_workspace.h ../memory_node.h | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=bgemm_outer_impl -c $< -o $@

$(VECTOR_OBJ): $(VECTOR_SRC) shared_workspace.h ../memory_node.h | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=bgemm_vector_impl -c $< -o $@

$(LIB_NAME): $(DISPATCH_OBJ) $(BACKEND_OBJS)
	$(CXX) $(COMMON_FLAGS) $(DISPATCH_OBJ) $(BACKEND_OBJS) $(LDFLAGS_SO) -o $@

$(FINAL_LIB): $(LIB_NAME) | $(KERNELS_DIR)
	cp -f $< $@

$(TEST_NAME): $(TEST_MAIN_OBJ) $(BACKEND_OBJS)
	$(CXX) $(COMMON_FLAGS) $(TEST_MAIN_OBJ) $(BACKEND_OBJS) $(LDFLAGS_EXE) $(MPI_LDFLAGS) -o $@

clean:
	rm -f $(LIB_NAME) $(TEST_NAME) $(FINAL_LIB)
	rm -rf $(BUILD_DIR)

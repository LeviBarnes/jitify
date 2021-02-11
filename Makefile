
DOXYGEN ?= doxygen
CXXFLAGS ?= -O3 -Wall -Wconversion -Wextra -Wshadow -g -fmessage-length=80

CXX11 ?= 1

CUDA_DIR ?= /opt/cuda-11.2/
CUDA_INC_DIR ?= $(CUDA_DIR)/include
NVCC = $(CUDA_DIR)/bin/nvcc

CXXFLAGS += -pthread

ifeq ($(CXX11),1)
	CXXFLAGS += -std=c++11
	NVFLAGS += -std=c++11
endif

ifeq ($(CXX14),1)
	CXXFLAGS += -std=c++14
	NVFLAGS += -std=c++14
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
	NVFLAGS += -g -G
endif
EMBED_BEGIN = -rdynamic -Wl,-b,binary,
EMBED_END   = ,-b,default

NVCC_EMBED_BEGIN = -rdynamic -Wl\,-b\,binary\,
NVCC_EMBED_END   = \,-b\,default

EMBED =
NVCC_EMBED =
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	# Embed a header into the executable (only supported by gcc on Linux)
	EMBED = $(EMBED_BEGIN)example_headers/my_header2.cuh$(EMBED_END)
	NVCC_EMBED = $(NVCC_EMBED_BEGIN)example_headers/my_header2.cuh$(NVCC_EMBED_END)
	CXXFLAGS += -D LINUX 
	CUDA_LIB_DIR = $(CUDA_DIR)/lib64
else ifeq ($(UNAME_S),Darwin)
	CUDA_LIB_DIR = $(CUDA_DIR)/lib
endif

INC += -I$(CUDA_INC_DIR)
INC += -I.
PROTEUS_INC = -I/slowfs/ltgpe_s1_us03/levi/build/saltM_Integ-current/proteus/proteuslib/
LIB += -ldl -L$(CUDA_LIB_DIR) -lcuda -lcudart -lnvrtc -lnvToolsExt

HEADERS = jitify.hpp \
          example_headers/my_header1.cuh.jit \
          example_headers/my_header2.cuh

all: jitify_example

JITIFY_EXAMPLE_DEFINES = -DCUDA_INC_DIR="\"$(CUDA_INC_DIR)\""

jitify_example: jitify_example.cpp $(HEADERS) K_test.cu.jit
	$(CXX) -o $@ $< $(CXXFLAGS) $(INC) $(LIB) $(EMBED) $(JITIFY_EXAMPLE_DEFINES)

simple_jit: simple_jit.cpp $(HEADERS) K_test.cu.jit
	$(CXX) -o $@ $< $(CXXFLAGS) $(INC) $(LIB) $(EMBED) $(JITIFY_EXAMPLE_DEFINES)

%.jit: % stringify
	./stringify $< > $@

stringify: stringify.cpp
	$(CXX) -o $@ $< -O3 -Wall

K_test_kernel.o: K_test.cu
	$(NVCC) -c -o $@ $< $(NVFLAGS) $(PROTEUS_INC) $(INC)

K_test:K_test_harness.cu  K_test_kernel.o
	$(NVCC) -o $@ $< K_test_kernel.o $(NVLFAGS) $(LIB)

get-deps:
	sudo apt-get update
	# CMake is needed to build gtest.
	sudo apt-get install -y cmake
.PHONY: get-deps

GTEST_DIR = googletest
GTEST_STATIC_LIB = $(GTEST_DIR)/build/googlemock/gtest/libgtest.a
$(GTEST_STATIC_LIB):
	rm -rf $(GTEST_DIR)
	git clone https://github.com/google/googletest.git $(GTEST_DIR)
	cd $(GTEST_DIR) && git checkout release-1.8.1 && rm -rf build && mkdir build && cd build && cmake .. && make -j8

GTEST_INC = -I$(GTEST_DIR)/googletest/include
GTEST_LIB = -L$(GTEST_DIR)/build/googlemock/gtest -lgtest -lgtest_main

CUB_DIR ?= /tmp/cub
CUB_HEADER = $(CUB_DIR)/cub/cub.cuh
$(CUB_HEADER):
	rm -rf $(CUB_DIR)
	git clone https://github.com/NVlabs/cub.git $(CUB_DIR)
	cd $(CUB_DIR) && git checkout v1.8.0

CUB_INC = -I$(CUB_DIR)
JITIFY_TEST_DEFINES = -DCUDA_INC_DIR="\"$(CUDA_INC_DIR)\"" -DCUB_DIR="\"$(CUB_DIR)\""

jitify_test: jitify_test.cu $(HEADERS) $(GTEST_STATIC_LIB) $(CUB_HEADER) Makefile
	# Link a 2nd compilation unit to ensure no multiple definition errors.
	echo "#include \"jitify.hpp\"\n#include \"example_headers/my_header1.cuh.jit\"" \
        > jitify_2nd_compilation_unit.cpp
	nvcc -o $@ $< jitify_2nd_compilation_unit.cpp -rdc=true -std=c++11 -O3 \
        -Xcompiler "$(CXXFLAGS) $(NVCC_EMBED) -pthread" \
        $(JITIFY_TEST_DEFINES) $(INC) $(GTEST_INC) $(CUB_INC) $(LIB) $(GTEST_LIB)

test: jitify_test
	./jitify_test
.PHONY: test

doc: jitify.hpp Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc

clean:
	rm -f stringify
	rm -f example_headers/*.jit
	rm -f jitify_example
	rm -f jitify_test
	rm -rf $(GTEST_DIR)
	rm -rf $(CUB_DIR)
	rm -f *.o
.PHONY: clean
#-I/slowfs/ltgpe_s1_us03/levi/build/saltM_Integ-current/proteus/proteuslib/

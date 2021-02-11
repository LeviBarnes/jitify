/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  Simple examples demonstrating different ways to load source code
    and call kernels.
 */

#define USE_JITIFY 1
#if USE_JITIFY
#define JITIFY_ENABLE_EMBEDDED_FILES 1
//#define JITIFY_PRINT_INSTANTIATION 1
//#define JITIFY_PRINT_SOURCE 1
//#define JITIFY_PRINT_LOG 1
//#define JITIFY_PRINT_PTX 1
//#define JITIFY_PRINT_LINKER_LOG 1
//#define JITIFY_PRINT_LAUNCH 1
#include "jitify.hpp"
#else
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#endif


//#include "example_headers/my_header1.cuh.jit"
//#ifdef LINUX  // Only supported by gcc on Linux (defined in Makefile)
//JITIFY_INCLUDE_EMBEDDED_FILE(example_headers_my_header2_cuh);
//#endif

#include <cassert>
#include <cmath>
#include <iostream>


#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      return false;                                                       \
    }                                                                     \
  } while (0)

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}

std::istream* file_callback(std::string filename, std::iostream& tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "example_headers/my_header4.cuh") {
    tmp_stream << "#pragma once\n"
                  "template<typename T>\n"
                  "T pointless_func(T x) {\n"
                  "	return x;\n"
                  "}\n";
    return &tmp_stream;
  } else {
    // Find this file through other mechanisms
    return 0;
  }
}

template <typename T>
bool test_mathlibs() {
  // Note: The name is specified first, followed by a newline, then the code
  // 
#include "K_test.cu.jit"

  // Compile the program for compute_20 with fmad // disabled.
  char *opts[] = {"--pre-include=stdint.h", "--use_fast_math", "-I" CUDA_INC_DIR, 
             "--gpu-architecture=compute_60"};  
#if USE_JITIFY
  using jitify::reflection::instance_of;
  using jitify::reflection::NonType;
  using jitify::reflection::reflect;
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  
  thread_local static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(
      K_test_cu,                          // Code string specified above
      0, 
      //{example_headers_my_header1_cuh},  // Code string generated by stringify
      {"--pre-include=stdint.h", "--use_fast_math", "-I" CUDA_INC_DIR, 
             "--gpu-architecture=compute_60"}
      );
      //,file_callback);
#else
  //Initialize CUDA context
  CHECK_CUDA(cuInit(0));  
  CUdevice cuDevice;  CHECK_CUDA(cuDeviceGet(&cuDevice, 0));  
  CUcontext context;  CHECK_CUDA(cuCtxCreate(&context, 0, cuDevice));  

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(    nvrtcCreateProgram(&prog,         // prog 
                      K_test_cu,         // buffer
                      "K_test.cu",    // name                       
                      0,             // numHeaders                       
                      NULL,          // headers                       
                      NULL));        // includeNames
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                4,     // numOptions                                                  
                opts); // options
  // Obtain compilation log from the program.  
  size_t logSize;  
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];  
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));  
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {    exit(1);  }
  // Obtain PTX from the program.  
  size_t ptxSize;  
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];  
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.  
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  // Load the generated PTX and get a handle to the SAXPY kernel.  
  CUmodule module;  
  CUfunction kernel;  
  CHECK_CUDA(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));  
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, "K_test"));
#endif
      

  // Allocate
  size_t n = 128*1024;
  size_t n_fields = 3;
  size_t n_scalars = 1;
  T *d_A, *d_B;
  T *d_C;
  cudaMalloc((void**)&d_A, n*sizeof(T));
  cudaMalloc((void**)&d_B, n*sizeof(T));
  cudaMalloc((void**)&d_C, n*sizeof(T));
  float* h_A = (float*)malloc(sizeof(T) * n);
  float* h_B = (float*)malloc(sizeof(T) * n);
  float* h_C = (float*)malloc(sizeof(T) * n);

  // Initialize data
  h_A[2] = 1.0/3; h_A[3] = 1.5;
  h_B[2] = 3.0; h_B[3] = 2.0;
  
  // Copy data
  //cudaHostRegister(h_A, n*sizeof(T), 0);
  cudaMemcpy(d_A, h_A, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n * sizeof(T), cudaMemcpyHostToDevice);

  // Create device pointers to data
  T** d_fields;
  T* d_scalars;
  cudaMalloc((void**)&d_scalars, n_scalars*sizeof(T));
  cudaMalloc((void**)&d_fields, n_fields*sizeof(T*));
  T** h_fields = (T**)malloc(sizeof(T*)*n_fields);
  T* h_scalars = (T*)malloc(sizeof(T)*n_scalars);
  h_fields[0] = d_A;
  h_fields[1] = d_B;
  h_fields[2] = d_C;
  cudaMemcpy(d_fields, h_fields, sizeof(T*)*n_fields, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scalars, h_scalars, sizeof(T*)*n_scalars, cudaMemcpyHostToDevice);
  

#if USE_JITIFY
  dim3 grid(128);
  dim3 block(32);
  CHECK_CUDA(program.kernel("K_test")
                 .instantiate()
                 .configure(grid, block)
                 .launch(d_fields, d_scalars, n));
#else
  void* arr[] = {reinterpret_cast<void*>(&d_fields),
               reinterpret_cast<void*>(&d_scalars),
               reinterpret_cast<void*>(&n) };
  cuLaunchKernel(kernel, 128, 1, 1, 32, 1, 1, 0, 0, arr, 0);
  CHECK_CUDA( cuModuleUnload(module) );
  CHECK_CUDA( cuCtxDestroy(context) );
#endif
  cudaMemcpy(h_C, d_C, n*sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_fields);
  cudaFree(d_scalars);
  free(h_fields);
  free(h_scalars);

 // std::cout << "(" << h_A[2] << ", " << h_A[3] << 
 //          ") * (" << h_B[2] << ", " << h_B[3] << 
 //         ") -> (" << h_C[2] << ", " << h_C[3] << ")" << std::endl;

  bool good =  are_close(h_A[2]*h_B[2] - h_A[3]*h_B[3], h_C[2]) &&
               are_close(h_A[2]*h_B[3] + h_A[3]*h_B[2], h_C[3]);
  //cudaHostUnregister(h_A);
  free(h_A);
  free(h_B);
  free(h_C);
  return good;
}
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
#define TEST_RESULT(result) (result ? "PASSED" : "FAILED")

  // Uncached
  bool test_mathlibs_result = test_mathlibs<float2>();
  cudaError_t err = cudaGetLastError();
  if (err) std::cerr << "CUDA Error " << err << ". " << cudaGetErrorString(err) << std::endl;

  // Cached
  test_mathlibs_result &= test_mathlibs<float2>();
  

  std::cout << "test_mathlibs            " << TEST_RESULT(test_mathlibs_result)
            << std::endl;

  return !test_mathlibs_result;
}

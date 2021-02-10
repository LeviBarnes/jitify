#include <iostream>
#include <stdlib.h>
__global__ void K_test(float ** fields, float const * scalars, size_t n);

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}
#define CUDA_CHECK(A) err=cudaGetLastError();if(err)std::cout<<"CUDA error on line "<<A<<". "<<cudaGetLastError()<<std::endl;
int main(void) {

  typedef float2 T;
  cudaError_t err;
  // Allocate
  size_t n = 128*1024;
  size_t n_fields = 32;
  size_t n_scalars = 32;
  T *d_A, *d_B;
  T *d_C;
  cudaMalloc((void**)&d_A, n*sizeof(T));
  cudaMalloc((void**)&d_B, n*sizeof(T));
  cudaMalloc((void**)&d_C, n*sizeof(T));
  float* h_A = (float*)malloc(sizeof(T) * n);
  float* h_B = (float*)malloc(sizeof(T) * n);
  float* h_C = (float*)malloc(sizeof(T) * n);
  // Initialize data
  for (size_t q=0; q<2*n;q++)
  {
     h_A[q] = rand()*1.0/RAND_MAX - 0.5;
     h_B[q] = rand()*1.0/RAND_MAX - 0.5;
  }

  
  // Copy data
  cudaMemcpy(d_A, h_A, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n * sizeof(T), cudaMemcpyHostToDevice);
  CUDA_CHECK(__LINE__);

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
  cudaMemcpy(d_scalars, h_scalars, sizeof(T)*n_scalars, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fields, h_fields, sizeof(T*)*n_fields, cudaMemcpyHostToDevice);
  CUDA_CHECK(__LINE__);
  
  dim3 grid(128);
  dim3 block(32);
  
  K_test<<<grid,block>>>((float**)d_fields, (float*)d_scalars, 2);
  
  cudaMemcpy(h_C, d_C, n*sizeof(T), cudaMemcpyDeviceToHost);
  CUDA_CHECK(__LINE__);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_fields);
  cudaFree(d_scalars);
  free(h_fields);
  free(h_scalars);
  CUDA_CHECK(__LINE__);

  std::cout << "(" << h_A[2] << ", " << h_A[3] << ") * (" << h_B[2] << ", " << h_B[3] << ") -> " << "(" << h_C[2] << ", " << h_C[3] << ")" << std::endl;

  bool good = are_close(h_A[2] * h_B[2] - h_A[3] * h_B[3], h_C[2]) &&
              are_close(h_A[2] * h_B[3] + h_A[3] * h_B[2], h_C[3]);
  if (good) std::cout << "Works." << std::endl;
  else std::cout << "No work." << std::endl;
   
  free(h_A);
  free(h_B);
  free(h_C);
  return !good;

}

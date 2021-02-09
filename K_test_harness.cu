#include <iostream>
__global__ void K_test(float * const * fields, float const * scalars, size_t n);

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}
int main(void) {

  typedef float2 T;
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
  
  // Copy data
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
  
  dim3 grid(128);
  dim3 block(32);
  
  K_test<<<grid,block>>>((float**)d_fields, (float*)d_scalars, n);
  
  cudaMemcpy(h_C, d_C, n*sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_fields);
  cudaFree(d_scalars);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_fields);
  free(h_scalars);

  std::cout << h_A[0] << " * " << h_B[0] << " -> " << h_C[0] << std::endl;

  bool good = are_close(h_A[0] * h_B[0] - h_A[1] * h_B[1], h_C[0]);
  if (good) std::cout << "Works." << std::endl;
  else std::cout << "No work." << std::endl;
   
  return !good;

}

#include <thrust/complex.h>

//namespace fops {
//  namespace builtins {
__global__ void K_test(float * const * fields, float const * scalars, size_t n)
{
  typedef float rscalar_t;
  typedef thrust::complex<rscalar_t> cscalarDev_t;
  // Complex multiply.
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(; tid < n; tid += blockDim.x * gridDim.x)
  {
    cscalarDev_t const a = ((const cscalarDev_t*) &fields[0])[tid];
    cscalarDev_t const b = ((const cscalarDev_t*) &fields[1])[tid];
    cscalarDev_t c = a * b;
    (cscalarDev_t &) fields[2][2*tid] = c;
  }
}


//}
//}}

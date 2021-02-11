//#include <thrust/complex.h>
template <typename T>
struct cmplx
{
  T x,y;
  __host__ __device__ cmplx(){};
  __host__ __device__ cmplx(T* in) {x = *in; y = *(in+1); }
  __host__ __device__ cmplx conj() {cmplx out = *this;out.y *=-1; return out;}
  __host__ __device__ T real() const { return x;}
  __host__ __device__ T imag() const { return y;}
  friend __host__ __device__  cmplx operator*(const cmplx A, const cmplx B) 
  {
    cmplx out; 
    out.x = A.x * B.x - A.y * B.y;
    out.y = A.x * B.y + A.y * B.x;
    return out;
  }

};

extern "C" __global__ void K_test(float ** fields, float const * scalars, size_t n)
{
  typedef float rscalar_t;
  typedef cmplx<float> cscalarDev_t;
  //typedef thrust::complex<rscalar_t> cscalarDev_t;
  // Complex multiply.
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(; tid < n; tid += blockDim.x * gridDim.x)
  {
    cscalarDev_t const a = ((cscalarDev_t*)fields[0])[tid];
    cscalarDev_t const b = ((cscalarDev_t*)fields[1])[tid];
    cscalarDev_t c = a * b;
    ((cscalarDev_t*) fields[2])[tid] = c;
  }
}


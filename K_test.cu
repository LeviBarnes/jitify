//#include <thrust/complex.h>
template <typename T>
struct cmplx
{
  T x,y;
  __device__ cmplx(){};
  __device__ cmplx(T* in) {x = *in; y = *(in+1); }
  __device__ cmplx(T r, T i) {x=r; y=i;}
  __device__ __inline__ cmplx conj() const {cmplx out = *this;out.y *=-1; return out;}
  __device__ __inline__ T real() const { return x;}
  __device__ __inline__ T imag() const { return y;}
  friend __device__  __inline__ cmplx operator*(const cmplx A, const cmplx B) 
  {
    cmplx out; 
    out.x = A.x * B.x - A.y * B.y;
    out.y = A.x * B.y + A.y * B.x;
    return out;
  }
  friend __device__ __inline__  cmplx operator+(const cmplx A, const cmplx B) 
  {
    cmplx out; 
    out.x = A.x + B.x;
    out.y = A.y + B.y;
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

#if 1
extern "C" __global__ void K_test2(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a + b;
    ((cscalarDev_t*) fields[2])[tid] = c;
  }
}

extern "C" __global__ void K_test3(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c(a.real() * b.real(), a.imag() * b.imag());
    ((cscalarDev_t*) fields[2])[tid] = c;
  }
}

extern "C" __global__ void K_test4(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test5(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test6(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test7(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test8(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test9(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

extern "C" __global__ void K_test10(float ** fields, float const * scalars, size_t n)
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
    cscalarDev_t c = a.conj() + b.conj();
    ((cscalarDev_t*) fields[1])[tid] = c;
  }
}

#endif

// Twelve conditioned FP64 Newton updates per selected line of sight.
// Build with nvcc -O3 --fmad=false -Xcompiler=-fPIC,-ffp-contract=off -shared
// and an -arch appropriate to the GPU. The caller owns the final ELISa potential test.
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <new>

namespace {
struct Ray { double x, y, z; };
struct Model { Ray line; double self, other, shift, rotation, offset; };
struct Buffers { Ray* base{}; double* t{}; size_t capacity{}; };

__device__ __host__ inline double dot(Ray a, Ray b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __host__ double stationary(Ray base, const Model& p) {
    double t = 0;
    const double limit = sqrt(dot(base, base)) / 4;
    const double projected_rotation = p.rotation*(p.line.x*p.line.x + p.line.y*p.line.y);
    for (int i=0; i<12; ++i) {
        Ray v{base.x+t*p.line.x, base.y+t*p.line.y, base.z+t*p.line.z};
        Ray w{v.x+p.shift, v.y, v.z};
        double vv=dot(v,v), ww=dot(w,w);
        double r=sqrt(vv), s=sqrt(ww);
        double a=p.self/(r*r*r), b=p.other/(s*s*s);
        Ray grad{a*v.x+b*w.x-p.rotation*v.x+p.offset, a*v.y+b*w.y-p.rotation*v.y, a*v.z+b*w.z};
        double along=dot(v,p.line), other_along=dot(w,p.line);
        double derivative=a*(1-3*along*along/vv) + b*(1-3*other_along*other_along/ww) - projected_rotation;
        if (!(derivative>0) || !isfinite(derivative)) return NAN;
        double step=dot(grad,p.line)/derivative;
        if (!isfinite(step)) return NAN;
        t -= fmin(limit, fmax(-limit, step));
    }
    return t;
}

__global__ void solve(const Ray* base, double* t, size_t n, Model p) {
    size_t i = static_cast<size_t>(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i<n) t[i]=stationary(base[i],p);
}
}
extern "C" void* eclipse_create() { return new (std::nothrow) Buffers(); }
extern "C" void eclipse_destroy(void* opaque) {
    auto* b = static_cast<Buffers*>(opaque);
    if (!b) return;
    if (b->base) cudaFree(b->base);
    if (b->t) cudaFree(b->t);
    delete b;
}
extern "C" int eclipse_solve(void* opaque, int gpu, size_t n, const double* input, const double* line,
                             double self, double other, double shift, double rotation, double offset, double* t) {
    static_assert(sizeof(Ray) == 3*sizeof(double));
    auto* b = static_cast<Buffers*>(opaque);
    const auto* base = reinterpret_cast<const Ray*>(input);
    Model p{{line[0],line[1],line[2]},self,other,shift,rotation,offset};
    if (!gpu) {
        for (size_t i=0;i<n;++i) t[i]=stationary(base[i],p);
        return 0;
    }
    if (!n) return 0;
    cudaError_t err;
    if (n>b->capacity) {
        if (b->base) { if ((err=cudaFree(b->base))!=cudaSuccess) return err; b->base=nullptr; }
        if (b->t) { if ((err=cudaFree(b->t))!=cudaSuccess) return err; b->t=nullptr; }
        b->capacity=0;
        if ((err=cudaMalloc(&b->base,n*sizeof(Ray)))!=cudaSuccess) return err;
        if ((err=cudaMalloc(&b->t,n*sizeof(double)))!=cudaSuccess) return err;
        b->capacity=n;
    }
    if ((err=cudaMemcpy(b->base,base,n*sizeof(Ray),cudaMemcpyHostToDevice))!=cudaSuccess) return err;
    solve<<<(n+127)/128,128>>>(b->base,b->t,n,p);
    if ((err=cudaGetLastError())!=cudaSuccess) return err;
    return cudaMemcpy(t,b->t,n*sizeof(double),cudaMemcpyDeviceToHost);
}

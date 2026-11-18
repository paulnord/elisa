// FP64 spectral quadrature for the opt-in native ELISa likelihood benchmark.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
constexpr int tile_rows = 256;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasGetStatusString(status));
}

struct DeviceArray {
    double* ptr = nullptr;
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    void allocate(size_t count) { check(cudaMalloc(&ptr, count * sizeof(double))); }
    void upload(const double* host, size_t count) {
        check(cudaMemcpy(ptr, host, count * sizeof(double), cudaMemcpyHostToDevice));
    }
};

struct Quadrature {
    cublasHandle_t handle = nullptr;
    int waves, bands;
    DeviceArray prefactor, exponent, weights, temperature, spectra, integral;
    Quadrature(int w, int b) : waves(w), bands(b) {}
    ~Quadrature() { if (handle) cublasDestroy(handle); }
};

__global__ void planck_table(const double* temperature, const double* prefactor,
                            const double* exponent, double* output, int waves, int cells) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cells) {
        const int w = idx % waves;
        output[idx] = prefactor[w] / expm1(exponent[w] / temperature[idx / waves]);
    }
}
}

extern "C" const char* spectra_error() { return last_error.c_str(); }

extern "C" void* spectra_create(int waves, int bands, const double* prefactor,
                                const double* exponent, const double* weights) {
    try {
        if (waves < 1 || waves > 16384 || bands < 1 || bands > 8)
            throw std::invalid_argument("Unsupported quadrature shape");
        auto q = std::make_unique<Quadrature>(waves, bands);
        check(cublasCreate(&q->handle));
        check(cublasSetMathMode(q->handle, CUBLAS_PEDANTIC_MATH));
        q->prefactor.allocate(waves);
        q->exponent.allocate(waves);
        q->weights.allocate(size_t(waves) * bands);
        q->temperature.allocate(tile_rows);
        q->spectra.allocate(size_t(waves) * tile_rows);
        q->integral.allocate(size_t(bands) * tile_rows);
        q->prefactor.upload(prefactor, waves);
        q->exponent.upload(exponent, waves);
        q->weights.upload(weights, size_t(waves) * bands);
        return q.release();
    } catch (const std::exception& e) {
        last_error = e.what();
        return nullptr;
    }
}

extern "C" int spectra_device(void* ctx, char* name, size_t size, int* runtime,
                              int* driver, int* blas) {
    try {
        int device;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&properties, device));
        std::snprintf(name, size, "%s", properties.name);
        check(cudaRuntimeGetVersion(runtime));
        check(cudaDriverGetVersion(driver));
        check(cublasGetVersion(static_cast<Quadrature*>(ctx)->handle, blas));
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

extern "C" int spectra_integrate(void* ctx, const double* temperature, int rows, double* output) {
    try {
        auto& q = *static_cast<Quadrature*>(ctx);
        const double one = 1.0, zero = 0.0;
        int starts = 0;
        for (int first = 0; first < rows; first += tile_rows) {
            const int count = std::min(tile_rows, rows - first);
            const int cells = count * q.waves;
            q.temperature.upload(temperature + first, count);
            planck_table<<<(cells + 255) / 256, 256>>>(q.temperature.ptr, q.prefactor.ptr,
                q.exponent.ptr, q.spectra.ptr, q.waves, cells);
            check(cudaGetLastError());
            // In column-major order the spectra are W x T and the weights W x B.
            // The resulting B x T has one contiguous passband vector per temperature.
            check(cublasDgemm(q.handle, CUBLAS_OP_T, CUBLAS_OP_N, q.bands, count, q.waves,
                &one, q.weights.ptr, q.waves, q.spectra.ptr, q.waves, &zero, q.integral.ptr, q.bands));
            check(cudaMemcpy(output + size_t(first)*q.bands, q.integral.ptr,
                size_t(count)*q.bands*sizeof(double), cudaMemcpyDeviceToHost));
            ++starts;
        }
        return starts;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

extern "C" void spectra_close(void* ctx) { delete static_cast<Quadrature*>(ctx); }

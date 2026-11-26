// Exact all-reference Bradley-Terry sums, without N x M intermediate arrays.
#include <cmath>
#include <cstdint>
#include <omp.h>
extern "C" void tau_factored(const double* a, const double* b, const float* w,
                             float* out, int64_t n, int64_t m, float s, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0; i<n; ++i) {
        double num=0, den=0;
#pragma omp simd reduction(+:num,den)
        for (int64_t j=0; j<m; ++j) {
            const double inv=1.0/(a[i]+b[j]);
            const double pos=a[i]*inv, neg=b[j]*inv;
            num += pos*w[j];
            den += (pos*neg)*w[j];
        }
        // Original uses means, so its epsilon scales by M in sum form.
        out[i]=s*(num/(den+1e-8*m));
    }
}
extern "C" void tau_fused(const float* a, const float* b, const float* w,
                          float* out, int64_t n, int64_t m, float s, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0; i<n; ++i) {
        double num=0, den=0;
#pragma omp simd reduction(+:num,den)
        for (int64_t j=0; j<m; ++j) {
            const float z=(a[i]-b[j])/s;
            const float e=std::exp(-std::abs(z));
            const float hi=1.0f/(1.0f+e), lo=e/(1.0f+e);
            const float p=z>=0 ? hi : lo;
            num += p*w[j];
            den += (hi*lo)*w[j];
        }
        out[i]=s*(num/(den+1e-8*m));
    }
}

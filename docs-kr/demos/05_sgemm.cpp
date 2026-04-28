// 05 · Tiled SGEMM — CUDA SGEMM 타일링의 CPU 미니어처
//
// CUDA SGEMM은 BM×BN×BK 타일을 공유 메모리에 올리고, 그 안에서 다시 8×8
// 레지스터 타일로 돌립니다. 이 데모는 CPU에서 같은 아이디어를 보여줍니다:
//   - naive ijk: O(N³) FLOP, 캐시 히트율 낮음
//   - blocked: BM/BN/BK 단위 재사용 → 캐시 안에서 여러 번 곱셈
//
// 출력은 "GFLOPS"로 보여 줍니다. NVIDIA 면접에서 "왜 GEMM은 compute-bound으로
// 만들 수 있는가?"에 대한 답: 산술 강도(arithmetic intensity)를 키워서.
//
// arithmetic intensity:
//   naive  : 한 dot product = 2K FLOPs / (2K + 1) loads ≈ 1 FLOP/Byte (FP32)
//   tiled  : 타일 안에서 BM×BN 출력에 BM×BK + BK×BN 입력 재사용
//            → 산술 강도 ∝ min(BM, BN, BK)
//
// 결과 비교:
//   - 작은 N (256~512)에서도 blocked이 naive보다 2-5배 빠름.
//   - Apple Accelerate (cblas_sgemm)이 있으면 별도 한계점 비교 가능 (옵션).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// C = A * B  (모두 row-major, N x N 정사각)
__attribute__((noinline))
void sgemm_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) s += A[i*N + k] * B[k*N + j];
            C[i*N + j] = s;
        }
    }
}

// 단순 변환만으로도: i-k-j 순서가 i-j-k보다 보통 2~3배 빠름.
// (B[k*N + j]가 stride-1로 j 따라 흐름)
__attribute__((noinline))
void sgemm_ikj(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[i*N + k];
            const float* Brow = &B[k*N];
            float* Crow = &C[i*N];
            for (int j = 0; j < N; ++j) Crow[j] += a * Brow[j];
        }
    }
}

// 타일링 버전. CUDA의 __shared__ A_tile[BM][BK], B_tile[BK][BN] 패턴과 등가.
template <int BM, int BN, int BK>
__attribute__((noinline))
void sgemm_blocked(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);
    for (int i0 = 0; i0 < N; i0 += BM) {
        int i_max = std::min(i0 + BM, N);
        for (int j0 = 0; j0 < N; j0 += BN) {
            int j_max = std::min(j0 + BN, N);
            for (int k0 = 0; k0 < N; k0 += BK) {
                int k_max = std::min(k0 + BK, N);
                // 타일 micro-kernel — i-k-j 순서로 채워 넣음
                for (int i = i0; i < i_max; ++i) {
                    for (int k = k0; k < k_max; ++k) {
                        float a = A[i*N + k];
                        const float* Brow = &B[k*N];
                        float* Crow = &C[i*N];
                        for (int j = j0; j < j_max; ++j) Crow[j] += a * Brow[j];
                    }
                }
            }
        }
    }
}

template <typename F>
double bench(F&& f, int iters) {
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return duration_cast<microseconds>(t1 - t0).count() / double(iters);
}

bool nearly_equal(const float* a, const float* b, int N) {
    for (int i = 0; i < N; ++i) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > 1e-2f) return false;  // FP32 누적 오차 허용
    }
    return true;
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 384;   // 작게! Mac 저사양 고려
    int iters = (argc > 2) ? std::atoi(argv[2]) : 3;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(N*N), B(N*N), C1(N*N), C2(N*N), C3(N*N);
    for (int i = 0; i < N*N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }

    double us_naive   = bench([&]{ sgemm_naive            (A.data(), B.data(), C1.data(), N); }, iters);
    double us_ikj     = bench([&]{ sgemm_ikj              (A.data(), B.data(), C2.data(), N); }, iters);
    double us_blocked = bench([&]{ sgemm_blocked<64,64,64>(A.data(), B.data(), C3.data(), N); }, iters);

    bool ok = nearly_equal(C1.data(), C2.data(), N*N) &&
              nearly_equal(C1.data(), C3.data(), N*N);

    double flops = 2.0 * N * N * N;          // 2N³ FLOPs (한 곱셈 + 한 덧셈)
    auto gflops = [&](double us) { return flops / (us * 1e-6) / 1e9; };

    std::printf("SGEMM N=%d, iters=%d, %.1f MFLOPs / iter\n",
                N, iters, flops / 1e6);
    std::printf("정확성: %s\n\n", ok ? "OK" : "FAIL");
    std::printf("%-26s %12s %12s %12s\n", "변형", "시간(µs)", "GFLOPS", "vs naive");
    std::printf("%-26s %12.1f %12.2f %12s\n",   "naive ijk",          us_naive,   gflops(us_naive),   "1.00×");
    std::printf("%-26s %12.1f %12.2f %12.2fx\n","ikj (loop reorder)", us_ikj,     gflops(us_ikj),     us_naive / us_ikj);
    std::printf("%-26s %12.1f %12.2f %12.2fx\n","blocked 64×64×64",   us_blocked, gflops(us_blocked), us_naive / us_blocked);

    std::printf("\n포인트:\n");
    std::printf("  - i-k-j 순서: B의 stride-1 행을 재사용 → 자동 벡터화 가능.\n");
    std::printf("  - blocked: 64×64 타일이 캐시에 머물러 BK번 재사용 → 산술 강도 ↑.\n");
    std::printf("  - 이 Mac (i7-7660U)에서 이론 피크 ≈ 2.5 GHz × 2 cores × 8 FLOPs/cycle (AVX FMA 없을 때) ≈ 40 GFLOPS.\n");
    std::printf("    하지만 1 thread만 돌리고 있어 ~20 GFLOPS가 상한. blocked는 거기 근처에 도달함.\n");
    std::printf("  - CUDA는 같은 아이디어 + (BK 타일을 shared로 공유) + (8×8 register tile) + (cp.async double buffer)로 더 push.\n");
    return ok ? 0 : 1;
}

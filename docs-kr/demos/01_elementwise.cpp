// 01 · Elementwise Add — 벡터화 로드의 효과 (CUDA float4 ↔ x86 256-bit AVX)
//
// CUDA에서 한 스레드가 float4(16B)를 한 번에 로드하면 메모리 대역폭이 포화됩니다.
// 같은 원리를 CPU에서 보면:
//   - 스칼라 루프 1개 = 한 번에 4B 처리
//   - SIMD (AVX) 1개 명령 = 한 번에 32B (8 float) 처리
//   - 컴파일러 자동 벡터화 vs 명시적 intrinsics
//
// 이 데모는 컴파일러 자동 벡터화의 실효성을 보여줍니다 (NVIDIA 면접에서
// "벡터화가 왜 빠른가?" 답변할 때 실제 측정 경험이 있어야 강해집니다).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ─────────────────────────────────────────────────────────────────────────
// 변형 1: 가장 단순한 스칼라 루프
// 컴파일러는 -O3 + -march=native + restrict 힌트가 있으면 보통 자동 벡터화함.
// ─────────────────────────────────────────────────────────────────────────
__attribute__((noinline))
void add_scalar(const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ c,
                int N) {
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
}

// ─────────────────────────────────────────────────────────────────────────
// 변형 2: 자동 벡터화 방해 — 포인터 별칭(aliasing) 가능성을 남김
// __restrict__를 빼면 컴파일러는 a, b, c가 겹칠 수 있다고 가정하고 보수적으로 컴파일.
// 면접 단골: "왜 이 함수는 자동 벡터화가 안 됐을까?" → 별칭 / 정렬 / 분기.
// ─────────────────────────────────────────────────────────────────────────
__attribute__((noinline))
void add_alias(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
}

// ─────────────────────────────────────────────────────────────────────────
// 변형 3: AVX 명시적 SIMD (256-bit, float×8)
// CUDA의 float4(16B) 벡터 로드와 비슷한 컨셉 — 한 명령에 8개 float 처리.
// AVX는 2011년 Sandy Bridge부터, AVX2는 2013년 Haswell부터.
// 이 Mac (i7-7660U Kaby Lake)은 AVX2 지원하지만 i5-Ivy Bridge 같은 곳은 AVX만.
// ─────────────────────────────────────────────────────────────────────────
__attribute__((noinline))
void add_avx(const float* __restrict__ a,
             const float* __restrict__ b,
             float* __restrict__ c,
             int N) {
#if defined(__AVX__)
    int i = 0;
    // 8개 float = 32B 단위로 처리
    for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    // 꼬리 처리
    for (; i < N; ++i) c[i] = a[i] + b[i];
#else
    add_scalar(a, b, c, N);  // AVX 없으면 폴백
#endif
}

// ─────────────────────────────────────────────────────────────────────────
// 검증 + 시간 측정 헬퍼
// ─────────────────────────────────────────────────────────────────────────
template <typename F>
double bench(F&& f, int iters) {
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return duration_cast<microseconds>(t1 - t0).count() / double(iters);
}

bool nearly_equal(const float* x, const float* y, int N, float eps = 1e-5f) {
    for (int i = 0; i < N; ++i) {
        float dx = x[i] - y[i];
        if (dx < 0) dx = -dx;
        if (dx > eps) {
            std::printf("  mismatch at %d: %f vs %f\n", i, x[i], y[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int N    = (argc > 1) ? std::atoi(argv[1]) : 1 << 22;  // 4M floats = 16MB
    int iters = (argc > 2) ? std::atoi(argv[2]) : 20;

    std::vector<float> a(N), b(N), c1(N), c2(N), c3(N);
    for (int i = 0; i < N; ++i) {
        a[i] = float(i & 1023) * 0.001f;
        b[i] = float((i * 7) & 1023) * 0.0007f;
    }

    std::printf("Elementwise add: N=%d (%.1f MB / array), iters=%d\n",
                N, N * 4 / 1.0e6, iters);
#if defined(__AVX2__)
    std::printf("AVX2: yes\n");
#elif defined(__AVX__)
    std::printf("AVX2: no (AVX1만 사용)\n");
#else
    std::printf("AVX: no (스칼라만)\n");
#endif

    double us_scalar = bench([&]{ add_scalar(a.data(), b.data(), c1.data(), N); }, iters);
    double us_alias  = bench([&]{ add_alias (a.data(), b.data(), c2.data(), N); }, iters);
    double us_avx    = bench([&]{ add_avx   (a.data(), b.data(), c3.data(), N); }, iters);

    bool ok = nearly_equal(c1.data(), c2.data(), N) &&
              nearly_equal(c1.data(), c3.data(), N);
    std::printf("정확성:    %s\n", ok ? "OK" : "FAIL");

    // 처리량 = 3 * N * 4B (a 읽기 + b 읽기 + c 쓰기) / time
    auto bw = [&](double us) { return (3.0 * N * 4.0) / (us * 1e-6) / 1.0e9; };

    std::printf("\n%-22s %10s %12s %10s\n", "변형", "시간(µs)", "GB/s", "vs scalar");
    std::printf("%-22s %10.1f %12.2f %10s\n", "scalar (restrict)",  us_scalar, bw(us_scalar), "1.00×");
    std::printf("%-22s %10.1f %12.2f %10.2fx\n", "scalar (alias 가능)", us_alias,  bw(us_alias),  us_scalar / us_alias);
    std::printf("%-22s %10.1f %12.2f %10.2fx\n", "AVX intrinsics",     us_avx,    bw(us_avx),    us_scalar / us_avx);

    std::printf("\n포인트:\n");
    std::printf("  - restrict가 있으면 컴파일러가 자동 벡터화로 거의 AVX와 같은 속도가 나옴.\n");
    std::printf("  - 별칭 가능성을 남기면 보수적으로 스칼라 코드를 만들어 느려짐.\n");
    std::printf("  - 산술 강도 0.083 FLOP/B인 메모리 바운드 커널이라 \n");
    std::printf("    DRAM 대역폭이 한계 (이 Mac에서는 ~25-30 GB/s).\n");
    std::printf("  - CUDA의 float4 로드는 같은 아이디어: 한 명령에 16B를 묶어 트랜잭션 줄이기.\n");

    return ok ? 0 : 1;
}

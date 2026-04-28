// 04 · Cache-Blocked Transpose — 캐시는 GPU의 공유메모리와 닮아 있다
//
// CUDA 행렬 전치는 "공유 메모리에 타일 단위로 로드 → 합체 쓰기" 패턴을 씁니다.
// CPU도 똑같은 이유로 타일링이 필요합니다:
//   - 메모리는 "한 번에 한 바이트"가 아니라 "한 번에 한 캐시라인(64B)"이 옵니다.
//   - 행 우선(row-major) 행렬에서 열 단위로 점프하면 매번 새 캐시라인을 가져와야 함.
//
// 변형:
//   1) naive_ji  — 출력 측에서 stride-1 (입력은 stride-N: 캐시 미스 폭발)
//   2) naive_ij  — 입력 측에서 stride-1 (출력은 stride-N)
//   3) blocked   — B×B 타일 단위로 처리 (입력/출력 둘 다 캐시 안에서 재사용)
//
// 면접: "왜 transpose가 느리죠?" → strided access → cache miss / TLB miss.
// → "어떻게 빨라지나?" → tile / blocking + 출력 측 일시 버퍼 (CUDA에선 shared mem).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// out[j*N + i] = in[i*N + j]   (정사각 N×N, in_layout = row-major)
__attribute__((noinline))
void transpose_naive_ij(const float* __restrict__ in, float* __restrict__ out, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            out[j * N + i] = in[i * N + j];   // 입력 stride-1, 출력 stride-N
}

__attribute__((noinline))
void transpose_naive_ji(const float* __restrict__ in, float* __restrict__ out, int N) {
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            out[j * N + i] = in[i * N + j];   // 출력 stride-1, 입력 stride-N
}

// 블록(타일) 전치 — 캐시 라인이 다 쓰이기 전엔 다시 같은 라인을 안 만지도록.
// CUDA의 __shared__ 타일 [BM][BN] + __syncthreads + 합체 쓰기 패턴의 CPU 등가물.
template <int B>
__attribute__((noinline))
void transpose_blocked(const float* __restrict__ in, float* __restrict__ out, int N) {
    for (int i0 = 0; i0 < N; i0 += B) {
        for (int j0 = 0; j0 < N; j0 += B) {
            int i_max = std::min(i0 + B, N);
            int j_max = std::min(j0 + B, N);
            for (int i = i0; i < i_max; ++i) {
                for (int j = j0; j < j_max; ++j) {
                    out[j * N + i] = in[i * N + j];
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

bool check(const float* a, const float* b, int N) {
    for (int i = 0; i < N * N; ++i) if (a[i] != b[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1024;   // 1024×1024 = 4MB
    int iters = (argc > 2) ? std::atoi(argv[2]) : 5;
    std::vector<float> in(N * N), o1(N * N, 0), o2(N * N, 0), o3(N * N, 0);
    for (int i = 0; i < N * N; ++i) in[i] = float(i & 1023);

    double us_ij  = bench([&]{ transpose_naive_ij  (in.data(), o1.data(), N); }, iters);
    double us_ji  = bench([&]{ transpose_naive_ji  (in.data(), o2.data(), N); }, iters);
    double us_b32 = bench([&]{ transpose_blocked<32>(in.data(), o3.data(), N); }, iters);

    bool ok = check(o1.data(), o2.data(), N) && check(o1.data(), o3.data(), N);

    auto bw = [&](double us) { return (2.0 * N * N * 4.0) / (us * 1e-6) / 1.0e9; };

    std::printf("Transpose %dx%d (%.1f MB), iters=%d\n", N, N, N*N*4/1.0e6, iters);
    std::printf("정확성: %s\n\n", ok ? "OK" : "FAIL");
    std::printf("%-30s %12s %12s %12s\n", "변형", "시간(µs)", "GB/s", "vs naive_ij");
    std::printf("%-30s %12.1f %12.2f %12s\n",   "naive ij (in: 1, out: N)", us_ij,  bw(us_ij),  "1.00×");
    std::printf("%-30s %12.1f %12.2f %12.2fx\n","naive ji (in: N, out: 1)", us_ji,  bw(us_ji),  us_ij / us_ji);
    std::printf("%-30s %12.1f %12.2f %12.2fx\n","blocked B=32",             us_b32, bw(us_b32), us_ij / us_b32);

    std::printf("\n포인트:\n");
    std::printf("  - row-major 메모리에서 stride-N 접근 = 매번 새 캐시라인 → 미스 폭발.\n");
    std::printf("  - blocked는 B×B 타일을 잠시 캐시에 가둬 두 방향 모두 stride-1처럼 동작.\n");
    std::printf("  - CUDA shared memory transpose: __shared__ float tile[32][33] (33은 bank conflict 회피).\n");
    std::printf("    공유 메모리 = 작은 의미의 'L1보다 더 가까운 캐시' + 협력적 사용.\n");
    return ok ? 0 : 1;
}

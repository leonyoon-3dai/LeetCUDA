// 02 · Parallel Reduction — 워프 셔플 트리의 CPU 버전
//
// CUDA의 `__shfl_xor_sync` 트리 reduce는 다음 두 트릭을 씁니다:
//   1) 여러 스레드가 부분합을 모은 뒤 하나로 합친다 (계층적 reduce)
//   2) 각 스레드가 "독립된" 누산기를 써서 메모리 충돌을 피한다
//
// CPU에서 같은 패턴:
//   - 각 thread가 자기 청크의 부분합을 로컬 변수에 누적 (cache-line 분리)
//   - 마지막에 한 번만 합쳐 atomic 또는 vector<>로 모음
// "공용 atomic 카운터" 같은 안티패턴과의 차이를 측정해 봅니다.
//
// NVIDIA 면접 단골: "reduce를 어떻게 병렬화하나요?" — 단순 atomic 답변은 감점,
// 계층적 reduce + 마지막에만 atomic이 정답.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ─────────────────────────────────────────────────────────────────────────
// 변형 1: 단일 스레드 — baseline
// ─────────────────────────────────────────────────────────────────────────
double reduce_serial(const double* x, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += x[i];
    return s;
}

// ─────────────────────────────────────────────────────────────────────────
// 변형 2: 여러 스레드가 "공용 atomic"에 직접 더함 — 안티패턴
// 모든 스레드가 같은 캐시라인에 RMW 경쟁 → 캐시 핑퐁으로 단일 스레드보다 느려짐.
// ─────────────────────────────────────────────────────────────────────────
// strict aliasing 위반 없이 비트 표현 그대로 옮기는 헬퍼 (C++20이면 std::bit_cast)
template <class To, class From>
static inline To bitcast(From from) {
    static_assert(sizeof(To) == sizeof(From), "size mismatch");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

double reduce_atomic_bad(const double* x, int N, int T) {
    std::atomic<long long> bits{0};  // double atomic 대신 비트 표현 누적
    auto worker = [&](int tid) {
        int chunk = (N + T - 1) / T;
        int lo = tid * chunk;
        int hi = std::min(lo + chunk, N);
        for (int i = lo; i < hi; ++i) {
            // 의도적 안티패턴: 모든 thread가 한 atomic을 RMW 경쟁
            long long old = bits.load(std::memory_order_relaxed);
            long long want;
            do {
                double cur = bitcast<double>(old) + x[i];
                want = bitcast<long long>(cur);
            } while (!bits.compare_exchange_weak(old, want, std::memory_order_relaxed));
        }
    };
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t) ths.emplace_back(worker, t);
    for (auto& th : ths) th.join();
    return bitcast<double>(bits.load());
}

// ─────────────────────────────────────────────────────────────────────────
// 변형 3: 계층적 reduce — 스레드별 로컬 누적 + 마지막 1회 합산 (정답 패턴)
// CUDA 워프 셔플 트리의 CPU 등가물.
// 핵심 1) per-thread 결과를 64B(캐시라인) 정렬해 false sharing 방지.
//      2) 마지막 합산은 단일 스레드에서.
// ─────────────────────────────────────────────────────────────────────────
struct alignas(64) PaddedDouble {
    double v;
    char pad[64 - sizeof(double)];
};

double reduce_hierarchical(const double* x, int N, int T) {
    std::vector<PaddedDouble> partial(T);
    auto worker = [&](int tid) {
        int chunk = (N + T - 1) / T;
        int lo = tid * chunk;
        int hi = std::min(lo + chunk, N);
        double s = 0.0;
        for (int i = lo; i < hi; ++i) s += x[i];
        partial[tid].v = s;
    };
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t) ths.emplace_back(worker, t);
    for (auto& th : ths) th.join();

    double total = 0.0;
    for (int t = 0; t < T; ++t) total += partial[t].v;
    return total;
}

template <typename F>
double bench(F&& f, int iters) {
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) (void)f();
    auto t1 = clk::now();
    return duration_cast<microseconds>(t1 - t0).count() / double(iters);
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1 << 22;
    int T     = (argc > 2) ? std::atoi(argv[2]) : (int)std::thread::hardware_concurrency();
    int iters = (argc > 3) ? std::atoi(argv[3]) : 5;
    if (T <= 0) T = 2;

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = double(i & 1023) * 0.0001 - 0.05;

    double truth = reduce_serial(x.data(), N);
    std::printf("Reduce: N=%d, threads=%d, iters=%d, sum=%.6f\n", N, T, iters, truth);

    double us_serial = bench([&]{ return reduce_serial(x.data(), N); }, iters);

    // atomic 안티패턴은 진짜 매우 느리므로 작은 입력으로만
    int Nsmall = std::min(N, 1 << 18);
    double us_atomic_per_elem = bench(
        [&]{ return reduce_atomic_bad(x.data(), Nsmall, T); }, 2) / Nsmall * N;

    double us_hier = bench([&]{ return reduce_hierarchical(x.data(), N, T); }, iters);

    double s_hier = reduce_hierarchical(x.data(), N, T);

    std::printf("\n%-32s %12s %12s\n", "변형", "시간(µs)", "vs serial");
    std::printf("%-32s %12.1f %12s\n", "serial",                us_serial,           "1.00×");
    std::printf("%-32s %12.1f %12.2fx (안티패턴!)\n",
                "shared atomic (CAS 루프)", us_atomic_per_elem,
                us_serial / us_atomic_per_elem);
    std::printf("%-32s %12.1f %12.2fx\n", "hierarchical (per-thread)",
                us_hier, us_serial / us_hier);

    std::printf("\n정확성: hier vs serial diff = %.3e\n", s_hier - truth);

    std::printf("\n포인트:\n");
    std::printf("  - 공용 atomic CAS 루프는 캐시 핑퐁으로 단일 스레드보다 수십~수백 배 느려질 수 있음.\n");
    std::printf("  - per-thread 누산 + 마지막에만 합치는 계층적 reduce가 NVIDIA 면접의 정답.\n");
    std::printf("  - PaddedDouble (alignas(64))로 false sharing을 차단함 — 다음 데모(06)에서 깊게 다룸.\n");
    std::printf("  - CUDA에서는 같은 아이디어를 워프 셔플(레지스터 트리) → 블록 reduce → 그리드 atomicAdd로 구현.\n");
    return 0;
}

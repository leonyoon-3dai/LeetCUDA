// 06 · False Sharing — CUDA bank conflict의 CPU 짝꿍
//
// 캐시 라인은 보통 64B. 두 스레드가 같은 라인에 있는 *다른* 변수를 각자 갱신하면
// MESI 프로토콜 때문에 라인 오너십이 핑퐁처럼 왔다 갔다 하면서 사실상 직렬화됩니다.
//
// 이게 CUDA 공유 메모리의 32-way bank conflict와 본질적으로 같은 현상:
//   - "물리적으로 같은 자원을 다른 사람들이 동시에 쓰려고 함" → 직렬화.
//
// 해결: alignas(64) 또는 패딩으로 캐시 라인 분리.
// NVIDIA 면접에서 false sharing은 멀티스레드 C++의 절대 단골 질문입니다.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

constexpr int ITERS = 5'000'000;

// 변형 1: 같은 캐시 라인에 카운터들을 욱여넣음 → false sharing
struct Packed {
    std::atomic<long> a{0};
    std::atomic<long> b{0};
    std::atomic<long> c{0};
    std::atomic<long> d{0};
};

// 변형 2: 64B 정렬로 분리 → 각 카운터가 자기 캐시 라인 차지
struct Padded {
    alignas(64) std::atomic<long> a{0};
    alignas(64) std::atomic<long> b{0};
    alignas(64) std::atomic<long> c{0};
    alignas(64) std::atomic<long> d{0};
};

template <typename S>
double run(S& s) {
    auto worker = [&](std::atomic<long>* p) {
        for (int i = 0; i < ITERS; ++i) p->fetch_add(1, std::memory_order_relaxed);
    };
    auto t0 = clk::now();
    std::thread t1(worker, &s.a);
    std::thread t2(worker, &s.b);
    std::thread t3(worker, &s.c);
    std::thread t4(worker, &s.d);
    t1.join(); t2.join(); t3.join(); t4.join();
    auto t1c = clk::now();
    return duration_cast<microseconds>(t1c - t0).count() / 1.0;
}

int main() {
    int hw = std::thread::hardware_concurrency();
    std::printf("False sharing demo  (hw_concurrency=%d, iters=%d/thread)\n", hw, ITERS);
    std::printf("Note: 이 Mac은 코어가 적어 효과가 다소 약할 수 있지만 분명히 보임.\n\n");

    Packed packed{};
    Padded padded{};
    double us_packed = run(packed);
    double us_padded = run(padded);

    std::printf("%-30s %12s %12s\n", "구조", "시간(µs)", "vs padded");
    std::printf("%-30s %12.0f %12.2fx\n", "packed (false sharing)", us_packed, us_packed / us_padded);
    std::printf("%-30s %12.0f %12s\n",    "padded (alignas(64))",   us_padded, "1.00×");

    std::printf("\n검증: 두 변형 모두 카운터 4개 × ITERS 만큼 증가했어야 함.\n");
    std::printf("  packed: %ld %ld %ld %ld\n",
                packed.a.load(), packed.b.load(), packed.c.load(), packed.d.load());
    std::printf("  padded: %ld %ld %ld %ld\n",
                padded.a.load(), padded.b.load(), padded.c.load(), padded.d.load());

    std::printf("\n포인트:\n");
    std::printf("  - 캐시 라인 = MESI 프로토콜의 단위. 라인을 공유하면 모든 RMW가 핑퐁.\n");
    std::printf("  - alignas(64) / padding 으로 라인 분리 = 면접 단골 답안.\n");
    std::printf("  - C++17이면 std::hardware_destructive_interference_size 사용 가능.\n");
    std::printf("  - CUDA bank conflict: 32 bank × 4B = 128B 영역에서 같은 bank를\n");
    std::printf("    여러 thread가 동시에 치면 직렬화. swizzle / padding 으로 회피.\n");
    return 0;
}

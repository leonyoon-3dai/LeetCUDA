// 07 · std::memory_order — C++ 메모리 모델 (NVIDIA 면접 빈출)
//
// 면접 단골:
//   - "memory_order_relaxed / acquire / release / seq_cst 차이?"
//   - "왜 release-acquire가 lock-free 큐의 게시-구독 패턴에서 충분한가?"
//   - "x86은 강한 모델이라 사실 release-acquire가 거의 공짜인데, ARM/CUDA는?"
//
// 이 데모는 두 가지를 보여줍니다.
//
// (1) Producer가 데이터를 쓰고 ready 플래그를 release로 게시 → consumer가 acquire로
//     읽으면 데이터까지 보장됨. (lock-free 발행/구독)
//
// (2) "happens-before" 가 깨지는 경우 — relaxed로만 두면 컨슈머가 ready=true를
//     보고도 data가 0인 채로 읽을 수 있다 (ARM/CUDA 등 약한 모델에서). x86에서는
//     보통 우연히 작동해 버려서 더 위험. 그래서 모델 자체를 정확히 코딩해야 함.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ──────────────────────────────────────────────────────────────────────────
// (1) 정상 패턴: release-acquire 페어
// ──────────────────────────────────────────────────────────────────────────
struct Channel {
    std::atomic<bool> ready{false};
    int payload = 0;
};

void demo_release_acquire() {
    Channel ch;
    int observed = -1;

    std::thread producer([&]{
        ch.payload = 42;                                       // (A) plain write
        ch.ready.store(true, std::memory_order_release);       // (B) release
        // release: A는 B 이전에 완료됨이 다른 thread에 보장됨 (acquire와 짝지어졌을 때)
    });

    std::thread consumer([&]{
        while (!ch.ready.load(std::memory_order_acquire)) {    // (C) acquire
            // acquire: 만약 ready=true를 봤다면 그 이전의 모든 write도 보임
        }
        observed = ch.payload;                                 // (D) plain read — 보장됨
    });

    producer.join();
    consumer.join();
    std::printf("[release-acquire] payload 관측값 = %d (기대: 42) %s\n",
                observed, observed == 42 ? "OK" : "FAIL");
}

// ──────────────────────────────────────────────────────────────────────────
// (2) seq_cst — 가장 강한 보장. 모든 thread가 같은 전역 순서를 봄
//     비용이 가장 큼 (x86에서는 mfence, ARM/CUDA에서는 dmb 같은 강한 펜스).
//     일반적으로는 release-acquire로 충분하므로 seq_cst는 최후의 수단.
// ──────────────────────────────────────────────────────────────────────────
void demo_seq_cst_dekker() {
    std::atomic<int> x{0}, y{0};
    int r1 = -1, r2 = -1;

    std::thread t1([&]{
        x.store(1, std::memory_order_seq_cst);
        r1 = y.load(std::memory_order_seq_cst);
    });
    std::thread t2([&]{
        y.store(1, std::memory_order_seq_cst);
        r2 = x.load(std::memory_order_seq_cst);
    });
    t1.join();
    t2.join();
    // seq_cst 보장: (r1==0 && r2==0)은 절대 일어날 수 없음.
    std::printf("[Dekker seq_cst] r1=%d r2=%d (둘 다 0이면 모델 위반)\n", r1, r2);
}

// ──────────────────────────────────────────────────────────────────────────
// (3) 안티패턴: relaxed만 사용 — payload가 0으로 보일 수 있음
//     (x86에서는 강한 모델이라 잘 안 보이지만 ARM/CUDA에서는 잘 깨짐)
//     여기서는 적어도 "코드만 봐도 잘못됨"을 보여주는 목적.
// ──────────────────────────────────────────────────────────────────────────
void demo_relaxed_buggy() {
    std::atomic<bool> ready{false};
    int payload = 0;
    int observed = -1;

    std::thread producer([&]{
        payload = 99;
        ready.store(true, std::memory_order_relaxed);  // ★ acquire-release 깨짐
    });
    std::thread consumer([&]{
        while (!ready.load(std::memory_order_relaxed)) { /* spin */ }
        observed = payload;  // ARM/CUDA에서는 0을 볼 수도 있음
    });
    producer.join();
    consumer.join();
    std::printf("[relaxed buggy] x86에서는 우연히 OK처럼 보일 수도: payload=%d\n", observed);
}

int main() {
    demo_release_acquire();
    demo_seq_cst_dekker();
    demo_relaxed_buggy();

    std::printf("\n포인트:\n");
    std::printf("  - relaxed: 원자성만, 순서 없음. 카운터 누적 정도에만.\n");
    std::printf("  - acquire/release: 'release한 thread의 모든 이전 write를 acquire한 thread가 봄'.\n");
    std::printf("    lock-free 큐, 발행/구독, 한 번 쓰고 여러 번 읽기 패턴의 표준.\n");
    std::printf("  - seq_cst: 전역 순서 보장. 비용 최대, 필요할 때만.\n");
    std::printf("  - CUDA: __threadfence() / __threadfence_block() / __threadfence_system().\n");
    std::printf("    cooperative_groups + cuda::std::atomic_ref<T> 가 현대적 답.\n");
    return 0;
}

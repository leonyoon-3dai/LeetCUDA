// 03 · Online Softmax — FlashAttention의 핵심 알고리즘
//
// 일반(safe) softmax는 3-패스가 필요합니다:
//   1) max = max_i x_i             — 스트림 1회 읽기
//   2) sum = sum_i exp(x_i - max)  — 스트림 1회 읽기
//   3) out_i = exp(x_i - max) / sum — 스트림 1회 읽기
//
// Online softmax는 1-패스로 (m, l) = (running max, running denom)을 유지합니다.
// 새 값 x가 들어왔을 때:
//   m_new = max(m, x)
//   l_new = l * exp(m - m_new) + exp(x - m_new)   ← "보정 인수 e^(m-m_new)"
//
// 두 청크를 합치는 법칙(병합):
//   m_AB = max(m_A, m_B)
//   l_AB = l_A * exp(m_A - m_AB) + l_B * exp(m_B - m_AB)
//
// 이 덕에 시퀀스를 K-V 청크 단위로 나눠 SRAM(공유메모리) 안에서 처리할 수 있어
// FlashAttention이 메모리 대역폭을 크게 절약합니다. NVIDIA 면접 단골.
//
// 이 데모는 1) 단순 3-패스, 2) online 1-패스의 결과가 정확히 같음을 검증하고
// 큰 입력에서 시간을 비교합니다.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ─────────────────────────────────────────────────────────────────────────
// 3-패스 (정석) softmax
// ─────────────────────────────────────────────────────────────────────────
void softmax_3pass(const float* x, float* y, int N) {
    float m = -INFINITY;
    for (int i = 0; i < N; ++i) m = std::fmax(m, x[i]);

    float l = 0.0f;
    for (int i = 0; i < N; ++i) l += std::exp(x[i] - m);

    float inv_l = 1.0f / l;
    for (int i = 0; i < N; ++i) y[i] = std::exp(x[i] - m) * inv_l;
}

// ─────────────────────────────────────────────────────────────────────────
// Online softmax: (m, l)을 유지하며 1-패스로 정규화 분모까지 계산
// 그 후 출력은 두 번째 패스로 한 번만 더 (이래도 2-패스라 메모리 트래픽 ↓).
// FlashAttention은 출력까지 한 번에 묶어서 정말 1-패스로 만듭니다.
// ─────────────────────────────────────────────────────────────────────────
void softmax_online(const float* x, float* y, int N) {
    float m = -INFINITY;
    float l = 0.0f;
    for (int i = 0; i < N; ++i) {
        float xi = x[i];
        if (xi > m) {
            // running max 갱신 — 기존 누적 분모 보정
            l = l * std::exp(m - xi) + 1.0f;
            m = xi;
        } else {
            l += std::exp(xi - m);
        }
    }
    float inv_l = 1.0f / l;
    for (int i = 0; i < N; ++i) y[i] = std::exp(x[i] - m) * inv_l;
}

// ─────────────────────────────────────────────────────────────────────────
// "병합 법칙" 직접 검증: 시퀀스를 두 청크로 쪼개 부분 (m, l)을 합쳐 본다
// FlashAttention은 K-V 시퀀스를 청크로 나눠 이 법칙을 반복 적용합니다.
// ─────────────────────────────────────────────────────────────────────────
struct ML { float m, l; };
ML stats(const float* x, int N) {
    float m = -INFINITY, l = 0.0f;
    for (int i = 0; i < N; ++i) {
        float xi = x[i];
        if (xi > m) { l = l * std::exp(m - xi) + 1.0f; m = xi; }
        else        { l += std::exp(xi - m); }
    }
    return {m, l};
}
ML merge(ML A, ML B) {
    float m = std::fmax(A.m, B.m);
    float l = A.l * std::exp(A.m - m) + B.l * std::exp(B.m - m);
    return {m, l};
}

template <typename F>
double bench(F&& f, int iters) {
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return duration_cast<microseconds>(t1 - t0).count() / double(iters);
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1 << 16;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 50;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    std::vector<float> x(N), y1(N), y2(N);
    for (int i = 0; i < N; ++i) x[i] = dist(rng);

    // 정확성 검증 (작은 N에서)
    softmax_3pass(x.data(), y1.data(), N);
    softmax_online(x.data(), y2.data(), N);
    float maxdiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = std::fabs(y1[i] - y2[i]);
        if (d > maxdiff) maxdiff = d;
    }

    // 병합 법칙 검증
    int half = N / 2;
    ML A = stats(x.data(),       half);
    ML B = stats(x.data() + half, N - half);
    ML AB = merge(A, B);
    ML full = stats(x.data(), N);
    float mdiff = std::fabs(AB.m - full.m);
    float ldiff = std::fabs(AB.l - full.l) / std::fmax(full.l, 1e-9f);

    double us_3p = bench([&]{ softmax_3pass (x.data(), y1.data(), N); }, iters);
    double us_on = bench([&]{ softmax_online(x.data(), y2.data(), N); }, iters);

    std::printf("Online softmax 검증: N=%d\n", N);
    std::printf("  max(|3pass - online|) = %.3e (FP32 한계 안)\n", maxdiff);
    std::printf("  병합법칙: m_diff=%.3e, l_rel_diff=%.3e\n\n", mdiff, ldiff);

    std::printf("%-22s %12s %12s\n", "변형", "시간(µs)", "vs 3-pass");
    std::printf("%-22s %12.1f %12s\n",  "3-pass softmax",  us_3p, "1.00×");
    std::printf("%-22s %12.1f %12.2fx\n", "online softmax", us_on, us_3p / us_on);

    std::printf("\n포인트:\n");
    std::printf("  - 3-패스는 입력을 3번 읽어야 함 → 메모리 바운드 (큰 N에서 패널티).\n");
    std::printf("  - Online은 1.x-패스로 같은 결과를 냄. 차이가 FP32 잡음 안에 들어옴.\n");
    std::printf("  - 청크별로 (m, l)을 모아 마지막에 merge() — FlashAttention의 본질.\n");
    std::printf("  - 면접 답: 'attention 행렬 S=QK^T를 풀로 띄우지 않고, K 청크를 SRAM에서\n");
    std::printf("    돌면서 online softmax로 누적 — HBM 트래픽 ↓, FLOPs는 같다.'\n");
    return (maxdiff < 1e-5f && ldiff < 1e-4f) ? 0 : 1;
}

# 13 · SGEMV / HGEMV — 행렬×벡터

> 원본 파일:
> - [`kernels/sgemv/sgemv.cu`](../../kernels/sgemv/sgemv.cu)
> - [`kernels/hgemv/hgemv.cu`](../../kernels/hgemv/hgemv.cu)
>
> **핵심 학습 포인트**:
> 1. GEMV는 GEMM과 달리 **산술 강도가 매우 낮음** (≈ 1 FLOP/B) — 완전 메모리 바운드.
> 2. **Warp-per-row** 매핑으로 한 워프가 한 행의 dot product를 담당.
> 3. K가 작을 때(K=16)는 **multiple rows per warp**로 스레드 낭비 방지.

---

## 1. 문제 정의 & 특성

$$y = A \cdot x, \quad A \in \mathbb{R}^{M\times K}, x \in \mathbb{R}^K, y \in \mathbb{R}^M$$

- **연산량**: `2 · M · K` FLOP (한 원소당 K FMA).
- **메모리**: A는 `M·K·4B`, x는 `K·4B`, y는 `M·4B`.
- **산술 강도**: `2MK / (MK·4 + K·4 + M·4) ≈ 0.5 FLOP/B` (M, K 큰 경우).

GEMM이 `O(N) FLOP/B`였던 것과 대조. GEMV는 **HBM 대역폭이 곧 성능 상한선**.

### 왜 재사용이 없는가

```
GEMM:  C[m][n] = Σ A[m][k] · B[k][n]
       A[m][k]는 C의 한 행 전체(N개)에 재사용
       B[k][n]은 C의 한 열 전체(M개)에 재사용

GEMV:  y[m] = Σ A[m][k] · x[k]
       A[m][k]는 y[m] 딱 1번만 사용
       x[k]만 M회 재사용
```

A는 거의 순수 스트리밍. 최적화 여지는 **x의 재사용**과 **메모리 합체**에 집중.

---

## 2. Warp-per-Row 매핑 (K = 32)

`sgemv.cu:32-52`의 `sgemv_k32_f32_kernel`:

```cuda
// 그리드: M/4 블록, 각 블록은 (32, 4) = 4 워프
// 워프 하나가 A의 한 행(K=32 원소)을 담당

int tx = threadIdx.x;         // 0..31 (lane)
int ty = threadIdx.y;         // 0..3  (워프 id)
int bx = blockIdx.x;
int lane = tx % WARP_SIZE;
int m = bx * blockDim.y + ty; // 이 워프가 담당하는 행 번호

if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;  // = 1 (K=32일 때)
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        int k = w * WARP_SIZE + lane;
        sum += a[m * K + k] * x[k];     // 각 레인이 1 원소 곱
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);   // warp 내 32원소 → 1
    if (lane == 0) y[m] = sum;
}
```

### 다이어그램

```
워프 (ty=0)        ─ 담당 행 m=bx*4   ─ A[m][0..31] × x[0..31]
워프 (ty=1)        ─ 담당 행 m=bx*4+1 ─ A[m][0..31] × x[0..31]
워프 (ty=2)        ─ 담당 행 m=bx*4+2
워프 (ty=3)        ─ 담당 행 m=bx*4+3

각 워프 내부:
  lane 0 → k=0:   A[m][0]  · x[0]
  lane 1 → k=1:   A[m][1]  · x[1]
  ...
  lane 31 → k=31: A[m][31] · x[31]
  ───────────────── warp_reduce_sum ─────────────────
  lane 0: 최종 sum → y[m]
```

### 왜 warp-per-row인가

- A의 한 행 32 원소 = 128 B = **1 합체 트랜잭션**.
- x의 32 원소 = 128 B = 역시 합체 (4 워프가 동일 x 공유 → L1 hit).
- warp reduce로 **SMEM 불필요**. `__shfl_xor`만으로 해결.
- 한 행이 한 워프에 완결 → **`__syncthreads()` 불필요**, occupancy 높게 유지.

---

## 3. K가 클 때 (K = 128, float4)

`sgemv.cu:58-83`의 `sgemv_k128_f32x4_kernel`:

```cuda
// 워프 하나가 K=128 담당, 레인이 4 원소씩 처리
int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;  // = 1 (K=128)

for (int w = 0; w < NUM_WARPS; ++w) {
    int k = (w * WARP_SIZE + lane) * 4;
    float4 reg_x = FLOAT4(x[k]);            // 16B 합체 로드
    float4 reg_a = FLOAT4(a[m * K + k]);
    sum += reg_a.x*reg_x.x + reg_a.y*reg_x.y + reg_a.z*reg_x.z + reg_a.w*reg_x.w;
}
sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
```

- 레인당 4 원소 → 워프 = 32 레인 × 4 = 128 원소 = K 전체.
- 로드는 `float4` 합체 → 4 × 128B = 512B per warp in 1 issue.

### K가 매우 큰 경우 (K=1024 등)

루프 반복이 늘어나며 한 워프가 K/128회 이터. 각 이터마다 `float4` 로드 + 4 FFMA + 누산. 산술 강도 개선 없이 반복만 늘어남 — **여전히 메모리 바운드**.

---

## 4. K가 매우 작을 때 (K=16)

`sgemv.cu:90-`의 `sgemv_k16_f32_kernel`.

### 문제

K=16이면 워프 32 레인 중 **절반만 일함**. 나머지 16 레인은 유휴 → 성능 반 토막.

### 해법: Multiple rows per warp

```
ROW_PER_WARP = 2 (K=16인 경우)
K_WARP_SIZE  = WARP_SIZE / ROW_PER_WARP = 16

워프 내 레인:
  lane 0..15  → 행 A (K 원소 0..15)
  lane 16..31 → 행 B (K 원소 0..15)
```

즉 **한 워프가 2개 행을 동시 처리**. 각 행은 16 레인이 담당.

```cuda
int k = lane % K_WARP_SIZE;  // 0..15
int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
//       └─ 같은 워프 내 lane < 16과 lane >= 16이 다른 m 가리킴

// warp_reduce_sum_f32<K_WARP_SIZE>(sum)  ← 16개만 reduce
// lane 0이 행 A의 y, lane 16이 행 B의 y를 저장
```

### 워프 sub-reduce

```cuda
warp_reduce_sum_f32<16>(sum)
// mask = 8, 4, 2, 1 (log2(16)=4 단계)
// lane 0~15 끼리만 합산, lane 16~31도 자기들끼리
```

**같은 워프 내 두 그룹이 독립적으로 reduce**. `__shfl_xor_sync`의 마스크가 16 이하라 서로 간섭 없음.

### 더 작은 K (K < 16)

`ROW_PER_WARP`를 4, 8로 늘려 계속 병렬도 확보. K=8이면 `ROW_PER_WARP=4`, 한 워프 = 4행.

---

## 5. HGEMV — FP16 변형

`kernels/hgemv/hgemv.cu`는 SGEMV와 동일 구조에서 타입만 half로 교체:

```cuda
half2 reg_x = HALF2(x[k]);
half2 reg_a = HALF2(a[m * K + k]);
// fp16 SIMD 곱셈 후 fp32로 승격해 누산
float partial = __half2float(reg_a.x) * __half2float(reg_x.x)
              + __half2float(reg_a.y) * __half2float(reg_x.y);
```

원칙은 [03-reduce](./03-reduce.md)와 같음: **저장/로드 fp16, 누산 fp32**.

---

## 6. 성능 상한선

A100 (HBM 2TB/s) 기준:

```
M=8192, K=8192, FP32:
  A 크기: 256 MB
  HBM에서 한 번 다 읽어오는 시간: 256 MB / 2 TB/s ≈ 128 μs
  FLOP: 2 · 8192² = 134 MFLOP
  → 달성 가능 최대: 134 MFLOP / 128μs ≈ 1.05 TFLOPS

cuBLAS sgemv 실측 (A100): 약 0.9 TFLOPS
  → HBM 대역폭의 ~86% 사용
  → **더 이상 최적화 여지 거의 없음**
```

그래서 LLM 추론에서 GEMV가 병목이면 보통:
1. **Batch GEMM (BGEMV)** 로 묶어서 산술 강도 높이기
2. **`torch.compile`/`flashinfer`** 등의 fuse
3. **양자화** (W4/W8)로 메모리 접근 바이트 감소

---

## 7. 요약

| K | 전략 | 쓰레드 활용 |
|---|------|-------------|
| K < 32 | Multiple rows per warp | 100% (ROW_PER_WARP 조정) |
| K = 32 | 1 워프 = 1 행, 1 원소/lane | 100% |
| K = 128 | 1 워프 = 1 행, float4 | 100%, 로드 효율 ↑ |
| K ≫ 128 | warp 내부 루프 | 100%, 여러 이터 |

---

## 다음 문서

👉 [14-embedding.md](./14-embedding.md) — 완전히 다른 메모리 패턴: **간접 gather**.

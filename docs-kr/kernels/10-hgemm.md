# 10 · HGEMM — WMMA / MMA Tensor Cores

> 원본 파일:
> - [`kernels/hgemm/wmma/hgemm_wmma.cu`](../../kernels/hgemm/wmma/hgemm_wmma.cu)
> - [`kernels/hgemm/mma/basic/`](../../kernels/hgemm/mma/basic/) (advanced)
>
> **핵심 학습 포인트**:
> 1. **Tensor Core란 무엇이고 왜 빠른가** — FFMA 64개를 1 명령에 수행.
> 2. **WMMA API** (C++ 고수준) vs **MMA PTX** (저수준, 최대 성능).
> 3. **Fragment**와 데이터 레이아웃 — 워프 32 스레드가 16×16×16 MMA를 분담하는 방식.

---

## 1. Tensor Core 개념

Volta 이후 NVIDIA GPU는 SM 내부에 **Tensor Core**라는 특수 행렬 곱셈 유닛을 탑재:

```
기존 CUDA Core (FFMA):
  c += a * b    (1 FLOP per 1 FFMA, 한 스레드당 1 이슈)

Tensor Core (1 MMA 명령):
  D[16×16] += A[16×16] · B[16×16]    (워프 32개 스레드가 함께 실행)
  = 16 × 16 × 16 × 2 = 8192 FLOP 전체, 1 명령 사이클 정도!
```

### 숫자로 비교 (A100, FP16)

- FP32 CUDA core: **19.5 TFLOPS**
- FP16 Tensor Core: **312 TFLOPS** (×16)
- BF16/TF32 Tensor Core: **156 TFLOPS**
- FP8 Tensor Core (H100+): **3958 TFLOPS**

**같은 클럭에서 16~32배 연산량**. GEMM, Conv, Attention의 대부분 계산이 Tensor Core로 이동.

---

## 2. Tensor Core의 "모양"

지원 행렬 크기(M×N×K):
- Volta: `16×16×16`, `32×8×16`, `8×32×16` (FP16 I/O, FP32 accum)
- Ampere: 위 + `16×8×16`, `16×8×8` 등 (BF16, TF32 지원)
- Hopper: WGMMA (warp-group MMA), `64×NxK` 등 대형 shape

### 16×16×16 시각화

```
      B (16×16)
     ┌─────────┐
     │         │
     │         │  K=16
     │         │
     └─────────┘
  A (16×16)      C (16×16) = A · B + C
 ┌────┐────────────┐
 │    │            │
 │    │            │  M=16
 │    │            │
 └────┘────────────┘
   K=16      N=16
```

- 입력 A, B: FP16 (16 × 16 = 256 원소 각각)
- 누산 C: FP16 **또는** FP32 (정밀도 선택 가능)
- **워프 32 스레드가 협력**하여 이 16×16×16 연산을 1개 명령에 처리.

### 워프 내 분담 (fp16 mma.m16n16k16)

A, B, C 원소는 32 스레드가 **고정된 레이아웃**으로 나눠 갖습니다. 예를 들어 스레드 0은 `A[0][0..15]`를 갖는 게 아니라, 특정한 스와핑 패턴으로 2~4개 원소를 가짐. 자세한 맵핑은 PTX ISA 문서에 있지만, **WMMA API는 이걸 은닉**.

---

## 3. WMMA API — 고수준 (권장 입문용)

`hgemm_wmma.cu:45-76`의 naïve WMMA 커널:

```cuda
#include <mma.h>
using namespace nvcuda;

template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
  const int load_gmem_a_m = blockIdx.y * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N;

  // ★ accumulator fragment — C[16][16] 영역을 워프 전체가 분담
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);

  for (int k = 0; k < K / WMMA_K; ++k) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;

    wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
    wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);  // ★ D = A·B + C
  }

  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n,
                          C_frag, N, wmma::mem_row_major);
}
```

### 라인별 의미

**`wmma::fragment<...>`**
- 워프가 공유하는 **추상적 행렬 조각**.
- 내부적으로 스레드 레지스터에 `__half a[8]` 같은 형식으로 저장.
- **자료형, 모양, 레이아웃을 템플릿 파라미터**로 전부 결정.

**`wmma::fill_fragment(C_frag, 0.0)`**
- 워프 32 스레드가 각자 할당된 C 원소를 0으로 초기화.

**`wmma::load_matrix_sync(A_frag, ptr, stride)`**
- `ptr[0..15, 0..15]` 영역을 A_frag에 로드.
- `stride` = 행 간격(N 또는 K) — row-major의 leading dimension.
- `_sync` 접미어: **워프 전원이 같이 호출**해야 함.

**`wmma::mma_sync(D, A, B, C)`**
- `D = A · B + C` 1개 Tensor Core 명령으로 실행.
- 이 한 줄이 **4096~8192 FLOP**을 한 사이클에 태움.

**`wmma::store_matrix_sync`**
- C_frag → 글로벌 메모리, `mem_row_major`로 저장.

### 이 커널의 한계

- 1 블록 = 1 워프(32 스레드) = 1 MMA 타일(16×16 C).
- **SMEM 재사용 없음** → A, B를 DRAM에서 매 k 이터마다 로드.
- 전형적 Block/Thread tile 최적화 전까지는 성능이 평범.

---

## 4. SMEM + Warp Tile WMMA — `hgemm_wmma_m16n16k16_mma4x2_kernel`

`hgemm_wmma.cu:79-`. 이 구조가 실전형:

```
BM = 16 × 4 = 64     (블록이 담당하는 M 방향 원소 수)
BN = 16 × 2 = 32     (블록이 담당하는 N 방향 원소 수)
BK = 16              (K 타일 크기)

워프 8개 (= 256 스레드) / 블록
각 워프는 자기 16×16 MMA 타일 담당

WMMA_TILE_M = 4  ← 블록 내 M 방향 워프 수
WMMA_TILE_N = 2  ← 블록 내 N 방향 워프 수
```

### 블록 내 워프 배치

```
워프 레이아웃 (8 워프 = 4 × 2):
  warp_id  warp_m, warp_n   담당 영역 (C 안에서)
    0       0, 0            [0..15][0..15]
    1       0, 1            [0..15][16..31]
    2       1, 0            [16..31][0..15]
    3       1, 1            [16..31][16..31]
    4       2, 0            [32..47][0..15]
    5       2, 1            [32..47][16..31]
    6       3, 0            [48..63][0..15]
    7       3, 1            [48..63][16..31]

블록 전체: C[0..63][0..31] = BM × BN = 64 × 32 영역
```

### 의사코드

```python
__shared__ half s_a[64][16], s_b[16][32]

for bk in range(K / 16):
    # 1. 256 스레드 협력 로드
    load_cooperatively s_a[64][16] and s_b[16][32] from HBM

    __syncthreads()

    # 2. 워프별로 자기 영역 MMA
    load A_frag from s_a[warp_m*16 .. warp_m*16+15][0..15]
    load B_frag from s_b[0..15][warp_n*16 .. warp_n*16+15]
    mma_sync(C_frag, A_frag, B_frag, C_frag)

    __syncthreads()

# 3. C_frag → global C 저장
store C_frag to C[blockIdx.y*64 + warp_m*16 .. ][blockIdx.x*32 + warp_n*16 .. ]
```

### 데이터 재사용

- `s_a[64][16]` 한 번 로드 → 워프 4개(warp_m 공유, warp_n 상이)가 **읽기 공유**.
- `s_b[16][32]` 한 번 로드 → 워프 2개가 공유.
- SMEM 크기: `64·16·2 + 16·32·2 = 2048 + 1024 = 3072 B = 3 KB` — 매우 여유.

---

## 5. MMA PTX — 저수준, 최대 성능

WMMA는 편리하지만 몇 가지 한계:
- fragment 레이아웃을 제어할 수 없음 → swizzle 최적화에 제약.
- 특정 비표준 shape (예: `mma.m16n8k16`) 는 WMMA가 직접 지원 안 함.

**MMA PTX**는 raw asm으로 Tensor Core 명령을 발사:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "      // D 4 레지스터
    "{%4, %5, %6, %7}, "      // A 4 레지스터 (8 half)
    "{%8, %9}, "               // B 2 레지스터 (4 half)
    "{%10, %11, %12, %13};\n"  // C 4 레지스터
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(ra0), ..., "f"(c0), ...
);
```

### 상세 해석

- `m16n8k16`: 작은 MMA shape. Ampere 이상에서 지원. A=16×16, B=16×8, C=16×8.
- `row.col`: A=row-major, B=col-major. (MMA 명령은 내부적으로 이런 방향을 요구.)
- `f32.f16.f16.f32`: D=fp32, A=fp16, B=fp16, C=fp32 누산.

### `ldmatrix` — MMA 전용 SMEM 로드

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(smem_ptr)
);
```

- SMEM의 `8×8` fp16 타일 4개를 32 스레드 레지스터에 **MMA가 요구하는 레이아웃**으로 한 번에 로드.
- 이 레이아웃이 복잡해서 일반 `LDS`로 구현하면 뱅크 충돌과 레지스터 재배치가 필요 — `ldmatrix`가 하드웨어로 해결.

### `mma` + `ldmatrix` + swizzle의 조합

LeetCUDA의 MMA 계열 커널(`hgemm/mma/basic/`, `hgemm/mma/swizzle/`)은 이 세 요소를 결합:

1. `cp.async` + swizzle SMEM 레이아웃으로 DRAM → SMEM 로드
2. `ldmatrix`로 SMEM → 레지스터 전송
3. `mma.sync` 명령 연속 발사 (한 이터 당 4~16개)
4. 레지스터 C_frag 누산
5. 저장 시 다시 swizzle 경로로 SMEM → DRAM

### 성능 비교 (참고 — 원 LeetCUDA 벤치)

M=N=K=4096, A100 기준:

| 구현 | TFLOPS | cuBLAS 대비 |
|------|--------|-------------|
| WMMA naïve | ~40 | 13% |
| WMMA + SMEM tile | ~180 | 58% |
| MMA + ldmatrix + cp.async | ~290 | 93% |
| MMA + swizzle + 다중 버퍼 | ~305 | 98% |
| cuBLAS hgemm | ~312 | 100% |

→ **정교한 MMA 커널이 cuBLAS의 98%에 도달**. 원저자의 HGEMM 프로젝트 자체가 이 경지를 다룹니다.

---

## 6. Hopper의 WGMMA (참고)

H100은 **warp-group MMA**를 도입:
- 128 스레드(4 워프) × 1 WGMMA 명령 = 64×N×K 전체 연산
- **비동기 MMA** (`wgmma.mma_async`): 이슈 후 다른 일 하다 wait
- **TMA** (Tensor Memory Accelerator): DRAM ↔ SMEM 비동기 복사, 1 스레드가 발사하여 전체 2D 타일 이동

LeetCUDA의 `kernels/hgemm/wgmma/` 가 이 방향.

---

## 7. 왜 Swizzle이 필수가 되는가

WMMA/MMA에서 `load_matrix_sync` 또는 `ldmatrix`가 SMEM 읽을 때, **각 스레드가 고정된 패턴으로 주소 분산 접근**을 합니다. 이 접근이 기본 SMEM 레이아웃에선 **8/16/32-way 뱅크 충돌**을 유발.

해결:
1. 행마다 `+8` 패딩 ([07-transpose.md](./07-transpose.md)에서 사용한 방식)
2. **XOR swizzle** — SMEM 용량 낭비 없이 충돌 제거 ([12-swizzle.md](./12-swizzle.md))

MMA 커널에서 2번이 표준.

---

## 8. 이 문서를 덮으며

- Tensor Core는 **딥러닝 시대의 SIMD**. 모든 프레임워크가 내부적으로 이를 타겟으로.
- WMMA로 시작해 MMA PTX로 진입하면 **가시적인 2~3배 성능 향상**.
- 다음 단계인 Flash Attention은 **MMA + Online Softmax + Tiling**의 종합예술.

---

## 다음 문서

👉 [11-flash-attn.md](./11-flash-attn.md) — Attention을 **SRAM에 다 담으면서** 긴 시퀀스로 스케일. 앞서 [05-softmax.md](./05-softmax.md)에서 배운 online softmax가 드디어 빛을 발합니다.

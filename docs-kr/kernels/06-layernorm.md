# 06 · Layer Norm & RMS Norm — 2-pass 통계

> 원본 파일:
> - [`kernels/layer-norm/layer_norm.cu`](../../kernels/layer-norm/layer_norm.cu)
> - [`kernels/rms-norm/rms_norm.cu`](../../kernels/rms-norm/rms_norm.cu)
>
> **핵심 학습 포인트**:
> 1. **평균과 분산을 두 번의 block reduce**로 계산.
> 2. `rsqrtf` 하드웨어 intrinsic으로 역제곱근 단일 명령.
> 3. `x`를 DRAM에서 한 번만 로드, 레지스터에 계속 살려두기.
> 4. RMS Norm = LayerNorm의 "평균 생략" 단순화. LLaMA 계열 표준.

---

## 1. 수식 복기

### Layer Norm
입력 행 `x` (길이 K):
$$
\mu = \frac{1}{K}\sum_i x_i, \quad \sigma^2 = \frac{1}{K}\sum_i (x_i - \mu)^2
$$
$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot g + b
$$

### RMS Norm
$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{K}\sum_i x_i^2}, \quad y_i = \frac{x_i}{\mathrm{RMS}(x) + \varepsilon'} \cdot g
$$

> RMS Norm은 **평균 계산을 생략**하여 LayerNorm을 가볍게 만든 변형. 성능/수렴에 큰 차이 없고 계산량이 적어 LLaMA, Qwen 등 현대 LLM이 표준 채택.

---

## 2. 실행 설정

둘 다 **per-token (= per-row)** 커널:

- 입력 `x` shape: `(N, K)` — N은 토큰 수(batch·seq_len), K는 hidden_size.
- 런치: `grid(N)`, `block(K)` (K ≤ 1024 제약).
- **한 블록이 한 행**을 처리하고, **블록 간 통신은 필요 없음**.

```
블록 0 → 행 0 의 mean/var/normalize 독립 수행
블록 1 → 행 1
...
블록 N-1 → 행 N-1
```

블록들이 완전히 독립이라 atomic이나 grid sync 불필요. **각 블록은 자체 reduce만** 수행.

---

## 3. Layer Norm — `layer_norm_f32_kernel`

`layer_norm.cu:54-78`:

```cuda
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y,
                                      float g, float b,
                                      int N, int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;   // 글로벌 위치
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;

  // ─── 1. 한 번만 로드, 이후 레지스터 재사용 ───
  float value = (idx < N * K) ? x[idx] : 0.0f;

  // ─── 2. 평균 ───
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  // ─── 3. 분산 (레지스터의 value 재활용) ───
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  // ─── 4. normalize + affine ───
  if (idx < N * K)
    y[idx] = ((value - s_mean) * s_variance) * g + b;
}
```

### 라인별 포인트

**`float value = x[idx]`** (`layer_norm.cu:64`)
- DRAM 접근은 **여기서 1번**. 이후 `value`는 레지스터에 생존.
- `x[idx]`를 `value - s_mean` 단계에서 다시 읽지 않기 위해 변수로 보관. 컴파일러도 보통 해주지만 **명시**가 안전.

**`block_reduce_sum_f32(value)`** (`layer_norm.cu:65`)
- [03-reduce](./03-reduce.md)에서 본 2-단계 reduce 헬퍼. 모든 스레드가 동일한 `sum`을 반환받음.

**`if (tid == 0) s_mean = sum / K; __syncthreads();`** (`layer_norm.cu:66-69`)
- 한 스레드가 평균을 SMEM에 기록.
- `__syncthreads()` 로 모든 스레드에게 평균값 가시화.

**`variance = (value - s_mean) * (value - s_mean)`** (`layer_norm.cu:70`)
- 각 스레드가 자기 원소의 편차 제곱을 계산.
- `value`는 아까의 레지스터 값, `s_mean`은 SMEM에서 한 번 읽음. **DRAM 재접근 없음.**

**`rsqrtf(variance / K + epsilon)`** (`layer_norm.cu:73`)
- `rsqrtf` = $1/\sqrt{x}$ 단일 PTX 명령 (`RSQRT.F32`).
- `1/std` 를 통째로 계산하므로 나눗셈 1회 절약.

**`((value - s_mean) * s_variance) * g + b`** (`layer_norm.cu:77`)
- affine 변환 포함. `g`, `b`는 scalar (원래는 shape(K) 벡터여야 하지만 여기선 간이 버전).
- 이 한 줄에 **FFMA 2개** (multiply+add 결합).

### 실행 타임라인 시각화

```
시간 →

t=0    : 각 스레드 x[idx] 로드 (합체)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━  (DRAM ~500 cycle)

t=500  : warp reduce (shfl_xor 5단계)  ━━━━
         ... block reduce                ━━
         s_mean 쓰기, __syncthreads()     ━

t=600  : variance 계산 (레지스터 only)    ━
         warp reduce + block reduce      ━━━━━━
         s_variance 쓰기, __syncthreads() ━

t=700  : normalize 계산 (레지스터 only)  ━
         STG.32 (y[idx])                ━━━━━━━━━━━━━━━━

              → 총 ~1000 cycle, 메모리 접근 2회(로드/스토어)
```

### 왜 fp32 버전인가

계산이 **분산(제곱합)** 을 포함해서, fp16으로 하면 언더/오버플로 위험이 큼. LLM은 보통:
- 저장: fp16/bf16
- LN 계산: fp32 cast 후 진행 (이 파일의 `bf16_f32` 변종 등)

---

## 4. float4 변형 — `layer_norm_f32x4_kernel`

`layer_norm.cu:84-120`. 한 스레드가 4 원소 담당:

```cuda
float4 reg_x = FLOAT4(x[idx]);
float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;  // local sum
float sum = block_reduce_sum_f32<NUM_THREADS>(value);
if (tid == 0) s_mean = sum / (float)K;
__syncthreads();

float4 reg_x_hat;
reg_x_hat.x = reg_x.x - s_mean;
... (4개)

float variance = reg_x_hat.x*reg_x_hat.x + ... + reg_x_hat.w*reg_x_hat.w;
variance = block_reduce_sum_f32<NUM_THREADS>(variance);
if (tid == 0) s_variance = rsqrtf(variance / K + epsilon);
__syncthreads();

float4 reg_y;
reg_y.x = reg_x_hat.x * s_variance * g + b;
... (4개)
FLOAT4(y[idx]) = reg_y;
```

### 이득

- **블록 스레드 수 1/4**(64개) → K가 1024인 경우에도 처리 가능.
- 메모리 트랜잭션 1/4, 레지스터 활용도 ↑.
- 단, `NUM_THREADS = 256/4 = 64`로 줄면 `NUM_WARPS = 2`, 2단계 reduce에서 워프 간 교환이 2개만 — 오버헤드 미미.

### K가 얼마나 커질 수 있나

`NUM_THREADS × 4 = K` 이므로 K=4096까지 한 블록에 담을 수 있음. 실제 LLM은 K가 4096~8192가 흔해서 **vec4 버전이 기본**.

---

## 5. RMS Norm — 더 단순한 버전

`rms_norm.cu:54-71`:

```cuda
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;

  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;                     // ★ 평균 없이 바로 x²
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  if (idx < N * K)
    y[idx] = (value * s_variance) * g;                // ★ 평균 빼기 없음
}
```

차이점 요약:

| 항목 | LayerNorm | RMSNorm |
|------|-----------|---------|
| Reduce 수 | **2회** (mean, var) | **1회** (sum of squares) |
| SMEM 사용 | `s_mean`, `s_variance` | `s_variance`만 |
| `__syncthreads()` 수 | 2회 | 1회 |
| affine | `· g + b` | `· g` (bias 없음) |

연산량 ~40% 감소. 학습 수렴 동등 (LLaMA 논문 및 후속 연구).

### 왜 "평균 빼기 없음"이 가능한가

LayerNorm은 **정확한 평균 제거**를 전제로 설계. 하지만 실제로는 입력 분포가 이미 평균 근처 0이면(특히 attention residual 후) 평균 빼는 의미가 작아짐. RMSNorm은 "어차피 평균은 거의 0이고, 중요한 건 스케일"이라는 경험적 관찰.

---

## 6. Welford 알고리즘 (참고)

LayerNorm은 2-pass지만, Welford 공식으로 1-pass 가능:

$$
\mu_{new} = \mu_{old} + \frac{x_n - \mu_{old}}{n}
$$
$$
M_{2,new} = M_{2,old} + (x_n - \mu_{old})(x_n - \mu_{new})
$$

이 갱신 규칙은 **숫자적으로 안정**적이지만 CUDA에서는:
1. 병합 연산자(online softmax처럼) 필요
2. 2-pass 구현이 이미 충분히 빠름 (입력은 레지스터에 있음)

그래서 실전 CUDA LayerNorm은 대부분 **2-pass 유지**. Welford는 주로 CPU/대용량 스트리밍에 사용.

---

## 7. 성능 비교 (개념적)

입력 (N=8192, K=4096), FP32 기준 상대값:

```
naive LayerNorm           ██████████████████  1.0×  (2-pass, scalar)
f32x4 LayerNorm           ████████████        0.65× (벡터화)
RMSNorm f32x4             █████████           0.50× (reduce 1회)
fused LayerNorm+GELU+...   LayerNorm 대비 ~0.7× 추가 여유
```

실제 LLM 학습/추론에서는 **LN 직전/직후 연산과 fuse**하는 것이 다음 최적화 단계. PyTorch/APEX의 fused LN이 이 방식.

---

## 8. 수치 안정성 노트

**`epsilon`** (`1e-5f`)
- 분산이 정확히 0인 경우 (상수 벡터) 0으로 나누기 방지.
- 너무 작으면 fp16 학습에서 inf, 너무 크면 정규화 효과 감소. 표준: 1e-5.

**`rsqrtf`의 정확도**
- PTX `RSQRT.F32`는 ULP 단위 근사. 일반 `1.0f / sqrtf(x)`는 fp32 정확하게 계산.
- LayerNorm 수준에서는 ULP 수준 오차가 **학습에 유의한 영향 없음** — intrinsic 사용이 표준.

**`bf16` 버전**
- 지수 범위는 fp32와 동일(8비트 지수) → 오버플로 걱정 ↓
- 가수 7비트로 매우 낮음 → 반드시 fp32 누산
- `rms_norm.cu` 뒷부분에 bf16 변형 존재. 핵심 구조는 f16과 동일, 로드만 `__nv_bfloat16 → float` 승격.

---

## 다음 문서

👉 [07-transpose.md](./07-transpose.md) — 메모리 접근 패턴의 또 다른 교과서 예제. **공유 메모리 패딩**으로 뱅크 충돌을 제거하는 기법.

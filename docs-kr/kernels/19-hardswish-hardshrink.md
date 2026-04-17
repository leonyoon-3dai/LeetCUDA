# 19 · HardSwish & HardShrink — 분기 활성화

> 원본 파일:
> - [`kernels/hardswish/hardswish.cu`](../../kernels/hardswish/hardswish.cu)
> - [`kernels/hardshrink/hardshrink.cu`](../../kernels/hardshrink/hardshrink.cu)
>
> **핵심 학습 포인트**:
> 1. **분기를 가진 활성화** 함수의 CUDA 구현 패턴.
> 2. `predicated mov` (SEL.F32) 로 워프 다이버전스 회피.
> 3. 모바일/임베디드 친화 활성화의 배경.

---

## 1. HardSwish

MobileNetV3가 제안한 **Swish의 부분선형 근사**:

```
HardSwish(x) = 0          if x ≤ -3
             = x          if x ≥  3
             = x · (x+3)/6  otherwise (구간 [-3, 3])
```

### 왜 "Hard"?

- 일반 Swish/SiLU = `x · σ(x)` — `expf` 필요. 모바일 NPU에서 비쌈.
- HardSwish는 **곱셈/덧셈/clamp만**으로 비슷한 곡선 흉내. 추론 인프라 친화적.

### 수치적 비교 (그래프 개념)

```
y
^
│         ╱─────  Swish
│        ╱
│      ╱── HardSwish (구간별)
│  __╱
│_/
└────────────────► x
  -3   0   3
```

두 곡선이 거의 일치, 미분도 부드러움(연속).

### 구현 — `hardswish.cu:33-41`

```cuda
__device__ __forceinline__ float hardswish(float x) {
  if (x >= 3.0f)         return x;
  else if (x <= -3.0f)   return 0.0f;
  else                   return x * (x + 3.0f) / 6.0f;
}
```

### 분기와 SEL 명령

세 분기처럼 보이지만 컴파일러는 보통 **predicated select**로 변환:

```ptx
SETP.GE.F32  P0, x, 3.0      // P0 = (x >= 3)
SETP.LE.F32  P1, x, -3.0     // P1 = (x <= -3)
FFMA.F32     mid, x, x, 3*x  // x*(x+3)/6 의 부분 계산
FMUL.F32     mid, mid, 1/6
SEL.F32      r, x,   mid, P0 // P0 면 x, 아니면 mid
SEL.F32      r, 0.0, r,   P1 // P1 면 0, 아니면 r
```

→ **실제 분기 명령(BRA) 없음**. 모든 스레드가 같은 명령 실행, predicate로 결과만 다름. **워프 다이버전스 회피**.

대신 양쪽 경로를 항상 계산하므로 **불필요한 ALU 발생**. 이 경우 산술이 매우 가벼워(곱셈 2~3개) 무시할 수 있음.

### `min(max(...))` 형태로 재작성 가능

```cuda
__device__ __forceinline__ float hardswish_v2(float x) {
  float clamped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
  return x * clamped / 6.0f;
}
```

분기 없이 동일 결과. PyTorch 공식 구현이 이 형태. 산술량 동일하지만 의도가 더 명확.

### FP16 변형 (`hardswish.cu:44-`)

`__hge`, `__hle`, `__hmul` 등의 fp16 intrinsic을 사용하지만 본질적으로 동일 구조. `__float2half(3.f)` 같은 상수 변환이 자주 등장.

---

## 2. HardShrink

```
HardShrink(x) = x   if |x| > λ
              = 0   otherwise           (보통 λ = 0.5)
```

### 용도

희소 표현(sparse representation), 노이즈 제거. 매우 작은 값을 0으로 죽여 "강한 신호"만 통과.

### 그래프 개념

```
y
^
│  ╱
│ ╱
│╱
0────·────► x
     ╲
      ╲
       ╲
            (구간 [-λ, λ] 에서 y=0)
```

### 구현 — `hardshrink.cu:35-41`

```cuda
__device__ __forceinline__ float hardshrink(float x) {
  if (x > LAMBD || x < -LAMBD)  return x;
  else                          return 0;
}
```

### 분기 제거 형태

```cuda
__device__ __forceinline__ float hardshrink_v2(float x) {
  float mask = (fabsf(x) > LAMBD) ? 1.0f : 0.0f;
  return x * mask;
}
```

또는 더 명시적으로:

```cuda
return (fabsf(x) > LAMBD) ? x : 0.0f;
```

컴파일러는 `SEL.F32` 한 명령으로 처리. 워프 다이버전스 없음.

---

## 3. 둘의 공통 구조

[01-elementwise](./01-elementwise.md), [02-activations](./02-activations.md) 와 100% 동일:

```
f32     - 1 스레드 1 원소
f32x4   - 1 스레드 4 원소 (FLOAT4)
f16     - 1 스레드 1 원소
f16x2   - 1 스레드 2 원소 (HALF2)
f16x8   - 1 스레드 8 원소
f16x8_pack - LDST128BITS 1회
```

각 변형의 코드 차이는 **로드/스토어 크기**와 **루프 펼침**뿐. 산술 본체는 위에서 본 `hardswish()` / `hardshrink()` 호출 1줄.

---

## 4. 분기 활성화의 일반 패턴

| 활성화 | 분기 수 | 분기 제거 가능? |
|--------|---------|----------------|
| ReLU | 0 (`fmax`) | N/A — 이미 분기 없음 |
| HardSwish | 3 | ✅ `min(max(x+3, 0), 6) · x / 6` |
| HardShrink | 2 | ✅ `(|x| > λ) ? x : 0` (predicate) |
| ELU | 2 | ⚠ `expf` 항상 계산 후 select |
| HardTanh | 3 | ✅ `min(max(x, -1), 1)` |
| LeakyReLU | 2 | ✅ `fmax(x, αx)` |

대부분의 "Hard" 활성화는 **clamp 형태로 재작성**해서 무분기로 만들 수 있습니다. 컴파일러가 자동 변환해주기도 하지만, 명시적으로 쓰는 것이 디버깅과 이식성에 유리.

---

## 5. 모바일/엣지 추론 맥락

이런 함수들이 만들어진 이유:

```
서버 GPU:  expf 비용 무시 가능 (Tensor Core 시대)
모바일 NPU: expf는 보통 별도 LUT(Look-Up Table) 또는 다항식 근사
            → 정확도 손실 또는 추가 메모리

해결: 처음부터 multiplicative-additive만 쓰는 활성화 설계
```

본 CUDA 구현은 호환성 차원에서 제공되는 것. 실제 GPU에서는 SiLU/GELU 가 일반적.

---

## 6. 마치며 — 활성화 함수 시리즈 정리

본 시리즈에서 다룬 활성화 (분포):

```
[02] ReLU, Sigmoid, GELU, Swish, ELU
[19] HardSwish, HardShrink
```

총 7종. 모두 **공통 구조 (벡터화 로드 + 산술 + 벡터화 스토어)** 의 변주. 일단 이 패턴을 익히면, 새로운 활성화(예: Mish, GLU, ReLU6 등)도 같은 골격에 산술만 갈아끼우면 됩니다.

---

## 다음 문서

본 시리즈의 마지막 문서로 갑니다.

👉 [99-others.md](./99-others.md) — 시리즈 전체 정리 + 미커버 항목 안내.

# 99 · 나머지 커널들 — 요약 가이드

> 앞선 12개 문서에서 다룬 개념(벡터화 로드 / warp reduce / SMEM 타일 / MMA / online softmax / swizzle)으로 **대부분의 나머지 커널이 설명**됩니다. 여기서는 각 커널의 **핵심 요점만** 정리합니다.

---

## SGEMV / HGEMV — 행렬×벡터

`kernels/sgemv/`, `kernels/hgemv/`.

$$y = A \cdot x, \quad A \in \mathbb{R}^{M\times K}, x \in \mathbb{R}^K$$

**특성**: 산술 강도 **1 FLOP/B 근처** — GEMM보다 훨씬 낮음. 완전 메모리 바운드.

**구현 요점**:
- 한 행을 한 워프가 담당 (warp-per-row) 또는 블록-per-row.
- 행 로드는 [03-reduce](./03-reduce.md) 패턴: 워프가 K개 원소를 나눠 로드 후 내적 + warp reduce.
- x는 전역 브로드캐스트 → **SMEM 또는 `__ldg`(read-only cache)** 활용.

**다이어그램**:
```
A (M × K):  행 m              × x (K)  =  y (M)
             └──── 워프 처리 ┘                  ★ 워프 reduce → y[m]
```

---

## Embedding

`kernels/embedding/`.

$$\text{out}[b, t] = W[\text{ids}[b, t]]$$

**특성**: 사실상 **간접 로드(gather)**. 합체 접근이 원천적으로 불가능한 경우가 많음.

**구현 요점**:
- 한 스레드가 한 토큰(한 임베딩 벡터) 담당.
- 벡터 내부는 연속 주소이므로 float4 합체 로드 가능.
- 중복 id가 있으면 L2 캐시에 걸리기 쉬움 → **배치 정렬**이 도움.

---

## RoPE (Rotary Position Embedding)

`kernels/rope/`.

주어진 hidden 벡터 `x`를 `cos, sin`과 회전 섞기.

```
for pair (x[2i], x[2i+1]):
    out[2i]   =  cos·x[2i]   - sin·x[2i+1]
    out[2i+1] =  sin·x[2i]   + cos·x[2i+1]
```

**특성**: 완전 elementwise. [01-elementwise](./01-elementwise.md)와 동일 구조 + **FMA 4개**.

**구현 요점**:
- cos, sin 테이블은 precomputed → constant cache / SMEM 올리기.
- 벡터화 로드 가능, 벡터화 쓰기 가능.

---

## NMS (Non-Maximum Suppression)

`kernels/nms/`.

객체 검출의 박스 필터. **상호 의존적 연산** — 단순 병렬화 어려움.

**구현 접근**:
- 블록당 대각 타일로 IoU 매트릭스 생성
- 비트마스크로 폐기 여부 표시
- CPU에서 최종 선별 (또는 prefix-scan)

→ 이 커널은 다른 것과 패턴이 많이 다름. 관심 있으면 [Torchvision NMS 구현](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu) 비교 권장.

---

## Transformer (attention + ffn 결합)

`kernels/transformer/`.

개별 블록(LN, QKV projection, attention, FFN, residual)을 **하나의 커널로 fuse**하려는 시도.

**왜 어려운가**: 각 서브블록이 서로 다른 병렬 축과 타일 크기를 선호 → 한 커널에 담으면 occupancy/레지스터 싸움.

**LeetCUDA 구현**: 교육용 버전. 프로덕션은 Flash Attention + 별도 LN + cuBLAS의 조합이 일반적.

---

## 활성화 추가본 (hardswish, hardshrink, elu)

`kernels/hardswish/`, `kernels/hardshrink/`, `kernels/elu/`.

모두 [02-activations](./02-activations.md)의 패턴:
- hardswish: `y = x · max(0, min(6, x+3))/6` — 분기 없음 (fmin/fmax).
- hardshrink: `y = (|x| > λ) ? x : 0` — predicated mov.
- elu: [02-activations §7](./02-activations.md) 참조.

모두 벡터화 로드 + 단순 ALU. fp16/fp32 변형이 기본.

---

## CUTLASS / Triton 디렉터리

`kernels/cutlass/`, `kernels/openai-triton/`.

직접 CUDA 작성 대신 **고수준 DSL**로 같은 GEMM을 표현하는 예시:
- **CUTLASS (C++)**: NVIDIA 공식. `Cute` 기반, fragment/swizzle/pipeline이 라이브러리화.
- **Triton (Python)**: 블록 단위 DSL, JIT 컴파일.

두 쪽 모두 **현대 고성능 커널 개발의 주류**. 직접 PTX를 짜는 대신 이런 DSL을 활용하는 것이 실무 트렌드.

---

## `others/`

`others/layer-norm.cu`, `others/mean.cu`, `others/reduce-v2.cu` 등 실험적/대안적 구현.

앞선 문서와 중복 또는 변형이 많아 개별 설명은 생략. 스스로 diff 해가며 읽으면 좋은 연습.

---

## WS-HGEMM (warp-specialized HGEMM)

`kernels/ws-hgemm/`.

**Hopper H100+** 의 새로운 패턴: 한 블록 안에서 워프마다 **역할을 분리** (일부는 producer = TMA/cp.async 로드 전담, 일부는 consumer = MMA 계산 전담). "Warp Specialization"이라 부르고, WGMMA·TMA와 조합해 초고성능.

개념 다이어그램:
```
Block (128 threads = 4 warps):
  warp 0: producer (TMA 로드 담당)
  warp 1: consumer (MMA 계산)
  warp 2: consumer (MMA 계산)
  warp 3: consumer (MMA 계산)

mbarrier (SMEM)로 동기화:
  producer: load → barrier arrive
  consumer: barrier wait → compute
```

이 구조는 FA-3, cuBLAS의 최신 구현이 모두 채택.

---

## 마무리

- **80% 이상의 커널**이 [01](./01-elementwise.md) ~ [06](./06-layernorm.md)의 패턴으로 설명됨.
- 고성능 영역은 [08](./08-sgemm.md), [10](./10-hgemm.md), [11](./11-flash-attn.md), [12](./12-swizzle.md)의 조합.
- 나머지는 이 레이어 위의 **응용** 혹은 **특수 도메인**(NMS, Embedding 등).

각 커널이 궁금해지면, 본 시리즈의 해당 패턴 문서로 먼저 돌아가 맥락을 잡고 → 원본 소스 코드를 직접 읽는 것을 권장합니다.

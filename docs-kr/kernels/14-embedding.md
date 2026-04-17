# 14 · Embedding — Gather 연산

> 원본 파일: [`kernels/embedding/embedding.cu`](../../kernels/embedding/embedding.cu)
>
> **핵심 학습 포인트**:
> 1. Embedding은 **간접 주소(indirect addressing)** 기반 gather — 합체 접근이 "토큰 내부"에서만 성립.
> 2. 블록 1개 = 토큰 1개 패턴으로 단순 병렬화.
> 3. vocabulary 크기에 비해 임베딩 차원이 클 때, L2/SMEM 재사용이 자연스럽게 발생.

---

## 1. 문제 정의

$$\text{output}[b, :] = W[\text{idx}[b], :]$$

- `idx`: shape `(N,)` 정수 인덱스.
- `W` (weight): shape `(V, E)`. V = vocabulary size, E = embedding dim.
- `output`: shape `(N, E)`. 각 토큰의 임베딩 벡터.

문자 그대로 **행 단위 복사**. 연산은 없고, 메모리 이동만 존재.

---

## 2. 메모리 접근 패턴

```
idx = [5, 2, 9, 0, ...]     (N 개의 정수)

W (V × E):
  ┌──────────── E ───────────┐
  │row 0                      │
  │row 1                      │
  │...                        │
  │row 5  ← idx[0]이 가리킴   │
  │...                        │
  │row 9  ← idx[2]이 가리킴   │
  │...                        │
  └──────────────────────────┘

output (N × E):
  row 0 ← W[5] 복사
  row 1 ← W[2] 복사
  row 2 ← W[9] 복사
  ...
```

**복사하는 행은 임의적**(idx 값에 따라). 하지만 **한 행 내부 E 원소**는 연속.

### 합체는 "토큰 내부"에서만

- 한 블록 = 한 토큰 → 그 블록 모든 스레드가 같은 행(W[idx[bx]])을 읽음 → ✅ 합체.
- 다른 블록 간 접근은 임의 → L2 캐시가 중복 idx에 도움.

---

## 3. 기본 구현

`embedding.cu:16-23`:

```cuda
__global__ void embedding_f32_kernel(const int *idx, float *weight,
                                     float *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;                  // ★ 블록 = 토큰
  int offset = idx[bx] * emb_size;      // ★ 담당 행의 시작 주소
  output[bx * emb_size + tx] = weight[offset + tx];
}
```

### 런치 설정

```
grid(N)      : 토큰 하나당 블록 하나
block(E)     : 임베딩 차원만큼 스레드 (E ≤ 1024 제약)
```

E가 1024를 넘으면 스레드가 여러 원소 담당하도록 루프 추가 필요.

### 동작 시각화

```
블록 bx = 0 (토큰 0):
  idx[0] 읽기 → offset 결정
  32 스레드(워프)가 weight[offset+0..31] 동시 접근 → ✅ 합체
  → output[0..31]에 저장 → ✅ 합체
```

---

## 4. 벡터화 변형 — `f32x4_pack`

`embedding.cu:36-45`:

```cuda
__global__ void embedding_f32x4_pack_kernel(...) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  // ★ 128b 한 번에 읽고 쓰기
  LDST128BITS(output[bx * emb_size + 4 * tx]) =
      LDST128BITS(weight[offset + 4 * tx]);
}
```

- 스레드 하나가 4 float = 16 B = 128 bit을 1 이슈로 복사.
- 단순 `float4 a = FLOAT4(src); FLOAT4(dst) = a;` 패턴.
- 블록 스레드 = E/4. E=1024라면 256 스레드/블록.

### 같은 테크닉을 half × 8로

`embedding.cu:56-69`의 `f16x8_kernel`은 `half` 8개 = 16 B를 한 번에 복사.

---

## 5. L2 캐시와 중복 idx

### 왜 L2가 도움 되나

```
배치에 같은 토큰이 여러 번 나오는 경우 (언어 데이터에 흔함):
  idx = [5, 5, 7, 2, 5, ...]
         └─ W[5] 한 번 읽으면 L2에 상주
            이후 idx=5인 블록들은 L2 hit으로 빠르게 처리
```

A100 L2 = 40MB. V=32K, E=768, fp16 embedding table:
- W 크기 = 32K × 768 × 2 B = **48 MB** — L2에 못 올림.

V=32K, E=128일 때 = 8 MB → L2에 완전 상주 가능.

### 입력 정렬의 팁

같은 idx들을 **가깝게 배치**하면 L2 hit 가능성 상승. 학습 단계에서 배치 셔플 시 완전 무작위 대신 "같은 sequence 묶기" 같은 전략이 때로 도움.

---

## 6. 큰 E 처리 (E > 1024)

블록당 최대 1024 스레드이므로 E=4096 같은 경우:

```cuda
// 각 스레드가 E/blockDim.x 원소 담당
for (int k = tx; k < emb_size; k += blockDim.x) {
    output[bx * emb_size + k] = weight[offset + k];
}
```

루프로 커버. 원본 파일의 변형들에서는 이런 경우를 `f16x8` 같은 넓은 벡터화로 처리.

---

## 7. 성능 상한선

E=4096, fp16, 배치 크기 B, 시퀀스 길이 S, N=B·S:

```
총 복사량 = N · E · 2 B = N · 8 KB

예: B=4, S=2048, N=8192 → 총 64 MB
A100 HBM 2 TB/s → ~32 μs 가 이론 하한
```

Embedding은 모델 전체에서 **매우 작은 비중**. 보통 총 실행 시간의 1% 미만.

---

## 8. Backward의 난제 (참고)

Forward는 단순 gather. Backward는 **scatter-add**:

```
dW[idx[b], :] += dOutput[b, :]
```

여러 b가 같은 idx를 가리키면 **같은 주소에 atomicAdd** → [04-dot-histogram](./04-dot-histogram.md)의 histogram 경합 문제가 재등장.

PyTorch 구현은 `torch.embedding_backward` 에 atomic 또는 sort+segment-reduce 방식 중 선택.

---

## 다음 문서

👉 [15-rope.md](./15-rope.md) — 더 흥미로운 elementwise 커널: 회전 위치 임베딩.

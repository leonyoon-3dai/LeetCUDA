# 17 · Transformer Block Fusion — 개념 가이드

> 원본 폴더: [`kernels/transformer/`](../../kernels/transformer/) (placeholder — 구현 대기 중)
>
> **핵심 학습 포인트**:
> 1. Transformer block을 구성하는 서브 커널들 조합.
> 2. **커널 퓨전(fusion)**의 경제학 — 어디까지 합칠까.
> 3. 앞서 본 12개 문서의 개념들이 **어떻게 하나의 블록에 수렴**하는지 조망.

---

## 1. Transformer Block 구조 복기

LLaMA 계열의 표준 디코더 블록:

```
          ┌──────── x ─────────┐
          │                     │
          ▼                     │
     RMSNorm (pre-norm)         │
          │                     │
          ▼                     │
     Wq·x, Wk·x, Wv·x           │  (3개 GEMM)
          │                     │
          ▼                     │
     RoPE(Q), RoPE(K)           │
          │                     │
          ▼                     │
   Flash Attention              │
          │                     │
          ▼                     │
     Wo · attn_out              │  (1 GEMM)
          │                     │
          ▼                     │
          + ─────── residual ───┘
          │
          ▼       ┌──── x' ────┐
    RMSNorm       │             │
          │       │             │
          ▼       │             │
  W_gate·x  W_up·x              │  (2 GEMM)
          │     │                │
     SiLU │     │                │
          ×─────┘                │
          │                     │
          ▼                     │
       W_down · ...              │  (1 GEMM)
          │                     │
          + ─────── residual ───┘
          │
          ▼
         output
```

한 블록 = **RMSNorm 2회, GEMM 7회, Attention 1회, RoPE 2회, SiLU 1회, residual 2회**.

---

## 2. 각 서브 커널이 앞 문서의 어디에 해당하는가

| 서브 | 본 시리즈 참조 |
|------|----------------|
| RMSNorm | [06-layernorm](./06-layernorm.md) |
| Wq/Wk/Wv/Wo 프로젝션 | [10-hgemm](./10-hgemm.md) (MMA) |
| RoPE | [15-rope](./15-rope.md) |
| Attention (QK^T, softmax, PV) | [11-flash-attn](./11-flash-attn.md) |
| W_gate, W_up, W_down (FFN) | [10-hgemm](./10-hgemm.md) |
| SiLU | [02-activations](./02-activations.md) — Swish |
| Residual (x + y) | [01-elementwise](./01-elementwise.md) |

즉 본 시리즈를 다 읽었다면 Transformer 블록의 **모든 구성요소**를 이미 이해한 셈.

---

## 3. 커널 퓨전의 경제학

각 서브가 별도 커널이면:
- **커널 런치 오버헤드** (수 μs × N개)
- **중간 텐서의 DRAM 왕복** (메모리 대역폭 낭비)
- 하지만 **튜닝 용이, 디버깅 쉬움**.

완전 퓨전이면:
- 오버헤드 제로
- 중간값이 SMEM/레지스터에 머무름 → 빠름
- 하지만 **레지스터/SMEM 압력** ↑, Occupancy ↓ 위험

### 실제 현대 구현의 관례

**항상 퓨전되는 것**:
- `RMSNorm + QKV projection` (정확히는 norm 후 3개 GEMM을 한 번에 묶은 `qkv_proj`)
- `SiLU · up_proj` (FFN 내부, SwiGLU 공식)
- Attention (FlashAttention 자체가 softmax 등을 내재화)

**보통 분리되는 것**:
- QKV projection vs RoPE: 중간 텐서를 그대로 쓸 수 있지만, 퓨전 시 복잡도 급증.
- Attention vs output projection.
- FFN의 gate/up/down 각각.

### 이유: **각 단계의 최적 파라미터가 다름**

```
RMSNorm:   블록당 1 토큰, 256~1024 threads
GEMM:      블록당 128×128 타일, 256 threads, BM/BN/BK 세팅
Attention: 블록당 Q 타일 64, 4 warps
FFN:       블록당 128×128 타일
```

서로 다른 그리드/블록 크기 → 한 커널에 담기 어려움.

---

## 4. 현대 고성능 구현이 실제로 하는 것

**vLLM / SGLang**:
- FusedSiLU: SiLU + element-wise mul
- Fused RMSNorm + Residual
- Paged Attention (FlashAttention 변형 + KV 캐시 관리)

**TensorRT-LLM**:
- Plugin으로 세밀한 퓨전. 모델별 그래프 최적화.

**Triton (torch.compile)**:
- 런타임에 **IR 수준에서 자동 퓨전** — pointwise 연산들을 가로지르는 fusion이 강점.

LeetCUDA의 `transformer/` 폴더는 **학습 차원의 전체 파이프라인 예시**를 담을 공간이지만 현재 구현이 비어있음. 관심 있으면 [vLLM/csrc/layernorm_kernels.cu](https://github.com/vllm-project/vllm) 같은 프로덕션 예제가 좋은 레퍼런스.

---

## 5. SwiGLU FFN 상세

LLaMA의 FFN은 기본 MLP가 아니라 **SwiGLU**:

$$
\text{FFN}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x))
$$

### 커널 구성

```
1. gate_out = W_gate · x         ← GEMM (hidden → intermediate)
2. up_out   = W_up · x           ← GEMM
3. act      = SiLU(gate_out) ⊙ up_out   ← 퓨전 가능 (elementwise)
4. out      = W_down · act       ← GEMM
```

3번은 elementwise이므로 1, 2의 출력과 합쳐 **GEMM + activation + 곱셈**으로 fuse 가능.

### Gated activation의 의미

`gate_out`이 "게이트" 역할, `up_out`이 "값". SiLU 활성화된 게이트가 값에 곱해져 **정보 흐름 제어**. ReLU MLP 대비 표현력 ↑.

---

## 6. KV 캐시 관점

추론(inference) 시 Transformer는 **autoregressive** — 이전 토큰의 K, V를 재계산하지 않고 **캐시**해둠.

```
Prefill 단계: 첫 프롬프트 (S 토큰) → K, V shape (B, H, S, D) 저장
Decode 단계: 1 토큰씩 생성, K, V에 append, attention은 이 전체를 봄
```

Decode 단계의 attention은 **Q shape (1, d), KV shape (S, d)** — 극도로 불균형. 전용 커널(PagedAttention, FlashAttention decode mode)이 따로 있음.

본 시리즈의 Flash Attention은 주로 **prefill/학습**용 패턴. Decode용 커널은 구조가 꽤 달라지며, 파라미터가 `B*H` 병렬, 짧은 Q가 특징.

---

## 7. 마치며

Transformer 블록을 만들 때 **어디를 퓨전할지**는 프로파일링에 기반한 결정:

1. 먼저 **분리된 커널들로 정확성 확보**.
2. `nvprof`/`ncu`로 커널 런치 오버헤드 비중 확인.
3. **DRAM 트래픽 큰 짝**을 우선 퓨전 (예: norm + proj).
4. 수치 정확도 재검증 (퓨전 시 **중간값 정밀도 선택**이 바뀌어 결과 살짝 변화 가능).

완전히 수동 퓨전을 하지 않고 **`torch.compile` + Triton**으로 자동화하는 것이 현재 트렌드.

---

## 다음 문서

👉 [18-ws-hgemm.md](./18-ws-hgemm.md) — Hopper H100의 신기술: **Warp Specialization + TMA + WGMMA**.

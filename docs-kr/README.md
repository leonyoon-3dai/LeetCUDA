# 📚 LeetCUDA 한국어 — NVIDIA C++ 면접 학습 키트

> [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)의 CUDA 커널 해설을 토대로, **NVIDIA C++ 직군 면접 대비**에 초점을 맞춘 한국어 학습 키트입니다.
>
> **NVIDIA GPU가 없는 Mac (저사양 Intel Mac 포함)** 에서도 핵심 개념을 *직접 빌드해서 숫자로 확인*할 수 있도록 설계되어 있습니다.

---

## 🚀 30초 빠른 시작

```bash
cd docs-kr/demos
make check        # 컴파일러/CPU 환경 점검 (가장 먼저!)
make run          # 8개 C++ 데모 빌드 → 실행
```

`make check`가 `xcrun: error...` 같은 메시지를 내면 macOS Command Line Tools가 망가진 상태입니다. 한 번만:

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install   # GUI 팝업에서 '설치' 클릭, 약 5–10분
```

→ 자세한 내용은 [`demos/README.md`](./demos/README.md) 참고.

---

## 🎯 이 학습 키트의 3개 트랙

| 트랙 | 폴더 | 무엇을 얻나 |
|------|------|------------|
| **A. 면접 로드맵** | [`nvidia-cpp/`](./nvidia-cpp/) | 6주짜리 C++ + 시스템 + GPU 사고법 마스터 플랜, 체크리스트 |
| **B. 실행 가능한 데모** | [`demos/`](./demos/) | NVIDIA GPU 없이 Mac에서 빌드되는 8개 C++ 데모 (CUDA 개념을 CPU로 매핑) |
| **C. 커널 정독** | [`kernels/`](./kernels/) | LeetCUDA 19개 CUDA 커널의 한국어 라인-바이-라인 해설 |

면접 준비라면 **A → B → C 순**이 효과적입니다. 우선 A에서 학습 지도를 잡고, B에서 손과 머리에 새기고, C에서 실제 GPU 코드의 깊이를 흡수.

---

## 🧪 트랙 B — Mac에서 돌아가는 8개 C++ 데모

각 데모는 단일 `.cpp` 파일, ≤200줄, 한국어 주석 포함. `make 01..08`로 개별 빌드, `make run`으로 일괄 실행.

| # | 파일 | CUDA 개념 매핑 | 면접 포인트 |
|---|------|-------------|----|
| 01 | [`elementwise`](./demos/01_elementwise.cpp) | `float4` 벡터 로드 | restrict / aliasing / AVX intrinsics |
| 02 | [`reduce`](./demos/02_reduce.cpp) | 워프 셔플 트리 | 계층적 reduce vs 공용 atomic 안티패턴 |
| 03 | [`softmax_online`](./demos/03_softmax_online.cpp) | FlashAttention 핵심 | (m, l) 병합 법칙 — 1-패스 softmax |
| 04 | [`transpose`](./demos/04_transpose.cpp) | shared mem 타일 전치 | 캐시 라인 / blocking / strided access |
| 05 | [`sgemm`](./demos/05_sgemm.cpp) | SGEMM BM/BN/BK 타일링 | 산술 강도 / loop reorder |
| 06 | [`false_sharing`](./demos/06_false_sharing.cpp) | bank conflict | MESI 핑퐁 / `alignas(64)` |
| 07 | [`memory_order`](./demos/07_memory_order.cpp) | `__threadfence` | release-acquire / seq_cst |
| 08 | [`move_rvo`](./demos/08_move_rvo.cpp) | (순수 C++) | RVO / move / forward / Rule of 5 |

---

## 🧠 트랙 A — NVIDIA C++ 면접 로드맵 (6주 가이드)

[`nvidia-cpp/README.md`](./nvidia-cpp/README.md) 의 핵심 골격:

```
1주차 ─ C++ 코어:        move/RVO, RAII, Rule of 5         → 데모 08
2주차 ─ 메모리 모델:     atomic, cache, false sharing       → 데모 06, 07
3주차 ─ 성능 사고:       Roofline, 산술 강도, 캐시 블로킹  → 데모 01, 04, 05
4주차 ─ 병렬 패턴:       reduce, scan, transpose, gemm     → 데모 02, 04, 05
5주차 ─ CUDA 사고법:     SIMT, 워프, shared, swizzle        → 00-cuda-basics + kernels/12
6주차 ─ 알고리즘 응용:   Online softmax, FlashAttention     → 데모 03 + kernels/11
```

면접 직전 한 페이지 체크리스트도 함께 제공.

---

## 📖 트랙 C — CUDA 커널 라인-바이-라인 (한국어)

CUDA 입문이라면 **반드시 [`00-cuda-basics`](./00-cuda-basics.md) 먼저** 읽으세요. 모든 후속 문서가 그 개념(스레드/워프/블록, 메모리 계층, SIMT)을 전제로 합니다.

| # | 문서 | 난이도 | 핵심 기법 |
|---|------|--------|----------|
| 00 | [CUDA 기초 개념](./00-cuda-basics.md) | ⭐ | Grid/Block/Thread, Warp, 메모리 계층, SIMT |
| 01 | [Elementwise & Vectorized Load (float4)](./kernels/01-elementwise.md) | ⭐ | `FLOAT4`, `LDST128BITS`, 메모리 대역폭 포화 |
| 02 | [활성화 함수 (ReLU / Sigmoid / GELU / Swish / ELU)](./kernels/02-activations.md) | ⭐ | 수치 안정성, `__expf`, FP16/FP32 혼용 |
| 03 | [Block All-Reduce — Warp Shuffle 트리](./kernels/03-reduce.md) | ⭐⭐ | `__shfl_xor_sync`, 2단계 reduce |
| 04 | [Dot Product & Histogram](./kernels/04-dot-histogram.md) | ⭐⭐ | atomicAdd, 공유 메모리 히스토그램 |
| 05 | [Softmax (Safe & Online)](./kernels/05-softmax.md) | ⭐⭐ | Max 빼기, Online Softmax 병합 법칙 |
| 06 | [Layer Norm & RMS Norm](./kernels/06-layernorm.md) | ⭐⭐ | 2-pass 통계, Welford(참고) |
| 07 | [Matrix Transpose & Bank Conflict](./kernels/07-transpose.md) | ⭐⭐ | Shared memory padding, 합체 접근 |
| 08 | [SGEMM — 타일링 & 레지스터 블로킹](./kernels/08-sgemm.md) | ⭐⭐⭐ | BM/BN/BK 타일, 8×8 레지스터 타일 |
| 09 | [SGEMM Async — Double Buffer & `cp.async`](./kernels/09-sgemm-async.md) | ⭐⭐⭐ | 파이프라이닝, Ampere 비동기 복사 |
| 10 | [HGEMM — WMMA / MMA Tensor Cores](./kernels/10-hgemm.md) | ⭐⭐⭐⭐ | `mma.m16n8k16`, PTX, Swizzle |
| 11 | [Flash Attention](./kernels/11-flash-attn.md) | ⭐⭐⭐⭐⭐ | Split-Q/KV, Online Softmax, SRAM 타일링 |
| 12 | [Swizzle — 뱅크 충돌 회피 패턴](./kernels/12-swizzle.md) | ⭐⭐⭐⭐ | XOR 기반 주소 재배치 |
| 13 | [SGEMV / HGEMV — 행렬×벡터](./kernels/13-sgemv-hgemv.md) | ⭐⭐ | Warp-per-row, K별 전략 |
| 14 | [Embedding — Gather 연산](./kernels/14-embedding.md) | ⭐ | 간접 주소, L2 재사용 |
| 15 | [RoPE — 회전 위치 임베딩](./kernels/15-rope.md) | ⭐⭐ | 쌍 단위 2D 회전, sin/cos 인라인 |
| 16 | [NMS — Non-Maximum Suppression](./kernels/16-nms.md) | ⭐⭐ | IoU 계산, 의존성 다루기 |
| 17 | [Transformer Block Fusion](./kernels/17-transformer.md) | ⭐⭐⭐ | 커널 합치기의 경제학 |
| 18 | [WS-HGEMM — Warp Specialization](./kernels/18-ws-hgemm.md) | ⭐⭐⭐⭐⭐ | Hopper TMA + WGMMA + CuTe |
| 19 | [HardSwish & HardShrink](./kernels/19-hardswish-hardshrink.md) | ⭐ | 분기 활성화, SEL 명령 |
| 99 | [나머지 커널 요약 가이드](./kernels/99-others.md) | — | 미커버 항목 간단 정리 |

면접 직전 우선순위: **08, 10, 11, 12** 4개 정독 + **00-cuda-basics**.

---

## 💻 실 GPU에서 돌려 보고 싶다면

이 저장소의 `kernels/*.cu` 자체는 NVIDIA GPU가 필요합니다. Mac 사용자도 무료로 쓸 수 있는 옵션:

- **Google Colab** — `Runtime → Change runtime type → T4 GPU` (무료 티어로 충분)
- **Kaggle Notebooks** — 무료 T4 ×2, 주당 30시간

```bash
# Colab 셀:
!nvidia-smi
!git clone https://github.com/xlite-dev/LeetCUDA && cd LeetCUDA && \
   pip install -e . && \
   python kernels/elementwise/elementwise.py
```

> 컴파일 절차 / Compute Capability 옵션은 루트 [`README.md`](../README.md) "Installation" 절 참고.

---

## ⚠️ 이 문서가 만들어진 환경

본 한국어 해설/데모 작성자는 **NVIDIA GPU가 없는 Intel Mac**에서 작성했습니다. 따라서:

- 커널 해설의 벤치마크 수치는 모두 **원 README / 원 커널 주석 인용**입니다.
- `docs-kr/demos/` 의 C++ 데모는 작성자가 직접 빌드 가능한 형태로 짠 *CPU 등가물* 입니다 (CUDA 코드 아님).
- 실 CUDA 검증이 필요한 경우 Colab/Kaggle 등 무료 GPU 환경을 권장.

---

## 🔗 참고 링크

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Flash Attention 논문](https://arxiv.org/abs/2205.14135)
- [Online Softmax 논문](https://arxiv.org/abs/1805.02867)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA의 GEMM 템플릿 라이브러리

## 📝 라이선스

원 프로젝트와 동일하게 **GPL v3.0**.

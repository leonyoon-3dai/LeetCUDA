# 📚 LeetCUDA 한국어 해설 (docs-kr)

> [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)의 CUDA 커널들을 **라인 바이 라인 한국어 주석**과 **Mermaid/ASCII 시각 자료**와 함께 학습하기 위한 문서 모음입니다.

## 📖 읽는 순서

CUDA 입문자라면 **반드시 [00-cuda-basics](./00-cuda-basics.md)를 먼저** 읽으세요. 모든 후속 문서가 이 문서의 개념(스레드/워프/블록, 메모리 계층, SIMT)을 전제로 합니다.

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
| 99 | [나머지 커널 요약 (sgemv, embedding, rope, nms 등)](./kernels/99-others.md) | — | 앞서 배운 패턴의 응용 |

## 🎯 이 문서가 지향하는 것

원저자의 코드에 대한 **한국어 주석과 시각 자료 레이어**를 얹는 것이 목적입니다. 코드 자체는 원저자([@xlite-dev](https://github.com/xlite-dev))의 저작물이며, 본 문서는:

- 각 커널의 **설계 의도**와 **왜 이렇게 짰는지**를 한국어로 설명
- **Mermaid 다이어그램**으로 메모리 접근 패턴 / 타일 구조 / 데이터 흐름 시각화
- **ASCII 아트**로 스레드 배치 / 공유 메모리 레이아웃 표현
- 벤치마크 수치는 원 README의 표/그래프를 인용 (본 작성자는 NVIDIA GPU 미보유)

## ⚠️ 실행 환경에 대한 주의

본 해설 작성자는 **NVIDIA GPU가 없는 환경**에서 문서를 작성했습니다. 따라서:

- 코드를 직접 실행하여 성능을 측정하지 않았습니다
- 벤치마크 수치는 모두 **원 README / 원 커널 주석**에서 인용한 것입니다
- 실제로 돌려볼 때는 CUDA 12.x + Compute Capability ≥ 8.0 (Ampere/Hopper) 권장

## 🔗 참고 링크

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Flash Attention 논문](https://arxiv.org/abs/2205.14135)
- [Online Softmax 논문](https://arxiv.org/abs/1805.02867)

## 📝 라이선스

원 프로젝트와 동일하게 **GPL v3.0**을 따릅니다.

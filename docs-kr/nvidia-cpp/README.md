# 🎯 NVIDIA C++ 면접 학습 로드맵 (한국어)

> 이 폴더는 NVIDIA의 C++ 중심 직군 (CUDA Library, cuDNN, cuBLAS, TensorRT, Driver, Compiler 팀 등) 면접에 자주 나오는 **C++ + 시스템 이해 + GPU 사고방식**을 한 곳에 정리한 학습 로드맵입니다.
>
> 이론만이 아니라 [`../demos/`](../demos/) 의 **실행 가능한 8개 C++ 데모**와 짝지어 학습하도록 설계했습니다. 각 주제 끝의 "📌 데모"가 실제로 빌드해서 숫자를 보는 단계입니다.

---

## 0. 학습 흐름 (4 트랙, 6주 권장)

```
1주차 ─ C++ 코어:        move/RVO/forwarding, RAII, Rule of 5  → 데모 08
2주차 ─ 메모리 모델:     atomic/memory_order, cache, false sharing → 데모 06, 07
3주차 ─ 성능 사고:       Roofline, 산술 강도, 캐시 블로킹       → 데모 01, 04, 05
4주차 ─ 병렬 패턴:       reduce, scan, transpose, gemm          → 데모 02, 04, 05
5주차 ─ CUDA 사고법:     SIMT, 워프, shared mem, bank conflict   → ../00-cuda-basics
6주차 ─ 알고리즘 응용:   Online softmax, FlashAttention 사고     → 데모 03, ../kernels/11
```

면접 1주 전부터는 **데모를 자기 말로 설명**하는 연습을 반복합니다. 코드를 외우는 게 아니라 "왜 이 줄이 있는가"를 자기 말로.

---

## 1. C++ 코어 — 무엇을 정확히 알고 있어야 하는가

### 1-1. 객체 라이프사이클: 5대 함수
| 함수 | 시그니처 | 언제 호출되나 |
|------|---------|--------------|
| 기본 생성자 | `T()` | `T x;` |
| 복사 생성자 | `T(const T&)` | `T y = x;`, by-value 인자 |
| 이동 생성자 | `T(T&&) noexcept` | `T y = std::move(x);`, 임시값 |
| 복사 대입 | `T& operator=(const T&)` | `y = x;` |
| 이동 대입 | `T& operator=(T&&) noexcept` | `y = std::move(x);` |

- **Rule of 0**: 자원 멤버를 RAII 타입(`std::unique_ptr`, `std::vector`)으로 두면 위 5개를 직접 짤 필요 없음. **이게 기본**.
- **Rule of 5**: 위 중 하나라도 직접 구현하면 5개 모두 직접 정의해야 함 (또는 `= default; = delete;`).
- **noexcept move**: 매우 중요. `std::vector`가 재할당할 때 noexcept move ctor가 있으면 *이동*, 없으면 *복사*하므로 성능 차이가 큼.

### 1-2. RVO / NRVO / copy elision
- **prvalue copy elision (C++17 이후 강제)**: `T f() { return T{}; }` — 호출처의 객체 자리에 *바로* 만든다. 복사도 이동도 *없음*.
- **NRVO (named local 반환)**: `T f() { T t; ...; return t; }` — 컴파일러가 가능하면 elide, 아니면 move로 fallback.
- **분기로 두 변수 중 반환**: 보통 elision 안 됨 → move.

### 1-3. `std::move` vs `std::forward`
- `std::move(x)` = `static_cast<T&&>(x)` — *runtime 작업 0*. 그냥 lvalue를 xvalue로 보이게 한다.
- `std::forward<T>(x)` — "forwarding reference" `T&&` 안에서 lvalue성/rvalue성을 *보존*하며 전달.
- 면접 함정: "std::move 자체가 데이터를 이동시킨다" — **틀림**. 데이터를 이동시키는 건 대상 클래스의 move ctor/op.

### 1-4. 템플릿 / 메타프로그래밍
필수만:
- 함수 템플릿, 클래스 템플릿, 부분/완전 특수화
- `if constexpr` (C++17) — 컴파일 타임 분기, dead branch 제거
- `decltype`, `std::declval`, `auto` 반환형 추론
- SFINAE (`std::enable_if`) — C++20 concepts로 더 깔끔해짐
- CRTP — "정적 다형성", 가상 함수 비용 없이 다형성 흉내

NVIDIA 코드(예: CUTLASS)는 *극한의 템플릿 메타프로그래밍*으로 구성되어 있습니다. CUTLASS 템플릿을 한 번이라도 읽어 봤다는 사실 자체가 강한 신호.

### 📌 데모
[`../demos/08_move_rvo.cpp`](../demos/08_move_rvo.cpp) — 인스트루먼트된 `Buffer` 클래스로 5가지 시나리오에서 복사/이동 횟수를 세어 본다.

---

## 2. 메모리 모델 — C++ atomic + 하드웨어 메모리 계층

### 2-1. `std::memory_order` 5종 + 1
| 순서 | 보장 | 비용 (x86) | 비용 (ARM/GPU) |
|------|------|-----------|----------------|
| `relaxed` | 원자성만 | 사실상 0 | 작음 |
| `consume` | (실무 deprecated; acquire로 대체) | — | — |
| `acquire` | "이후 read/write가 이 load 이전으로 못 감" | 0 (x86은 자동) | 펜스 필요 |
| `release` | "이전 read/write가 이 store 이후로 못 감" | 0 (x86은 자동) | 펜스 필요 |
| `acq_rel` | RMW용 — release+acquire | 0 (x86) | 펜스 필요 |
| `seq_cst` | 전역 순서까지 보장 | mfence | 강한 펜스 |

**핵심 패턴**: producer가 데이터 쓰고 ready 플래그를 `release`로 store, consumer가 ready를 `acquire`로 load — 이때 데이터가 정합적임이 보장. **lock-free 큐, 발행/구독, 이중 체크 lock 등 모든 게 이 위에 짓는다.**

### 2-2. CPU 캐시
- 캐시 라인 = 보통 64B (Intel/AMD), 128B (Apple Silicon).
- L1 ~32KB / core, L2 ~256KB-1MB / core, L3 공유.
- **MESI** 프로토콜 — 라인을 누가 가지고 있느냐에 따라 Modified/Exclusive/Shared/Invalid 상태.
- **False sharing**: 두 코어가 같은 라인의 다른 바이트를 갱신하면 라인 오너십 핑퐁 → 직렬화.
- 해결: `alignas(64)` / `std::hardware_destructive_interference_size` (C++17).

### 2-3. CUDA 메모리 계층 (간단 비교)
| 단위 | CPU 등가물 | 크기 | 지연 (대략) |
|------|-----------|------|------------|
| 레지스터 | 레지스터 | 256 / thread | 1 cycle |
| Shared memory | L1 (스마트) | 48-228KB / SM | ~30 cycle |
| L1/L2 cache | L2/L3 | MB | ~200 cycle |
| Global (HBM) | DRAM | GB | ~500+ cycle |

면접 단골: "shared memory와 L1 cache의 차이?" — shared는 *프로그래머가 명시적으로 관리*, L1은 하드웨어가 자동으로 관리. shared의 장점: 협력적 사용 (`__syncthreads` + 32 banks). 단점: 직접 swizzle/패딩으로 bank conflict 피해야 함.

### 📌 데모
- [`../demos/06_false_sharing.cpp`](../demos/06_false_sharing.cpp) — alignas 유무로 핑퐁 효과 측정.
- [`../demos/07_memory_order.cpp`](../demos/07_memory_order.cpp) — release-acquire 페어가 지키는 happens-before.

---

## 3. 성능 사고법 — Roofline & 산술 강도

### 3-1. Roofline 모델
한 커널의 도달 가능한 GFLOPS 상한:
```
GFLOPS_max = min( PEAK_FLOPS,  ARITHMETIC_INTENSITY × PEAK_BANDWIDTH )
```
여기서 **arithmetic intensity** = FLOPs / Bytes (DRAM에서 읽어 오는 바이트).

### 3-2. 대표 커널 분류
| 커널 | 산술 강도 (FP32) | 분류 |
|------|----------------|------|
| Elementwise add (`a+b`) | 0.083 FLOP/B | 메모리 바운드 |
| Reduce sum | ~0.25 FLOP/B | 메모리 바운드 |
| Softmax | ~0.5 FLOP/B | 메모리 바운드 |
| GEMM (큰 N) | ~N/2 FLOP/B (재사용) | **계산 바운드** |
| Attention (no fusion) | ~0.3 FLOP/B | 메모리 바운드 |
| Flash Attention | ~5+ FLOP/B | 더 계산 쪽 |

**원칙**: 메모리 바운드 커널을 빠르게 하는 길은 *FLOPs 줄이기 ≠ 답*. *바이트 줄이기*. = 데이터 재사용 = 타일링/캐싱.

### 3-3. CPU에서도 동일
GEMM이 compute-bound가 *되는* 이유는 BM×BN 출력 타일에 BM×BK + BK×BN 입력을 재사용하기 때문. 같은 입력을 캐시 안에서 BK번 곱셈에 쓴다 → 산술 강도 ∝ BK.

### 📌 데모
- [`../demos/01_elementwise.cpp`](../demos/01_elementwise.cpp) — 메모리 바운드의 한계.
- [`../demos/05_sgemm.cpp`](../demos/05_sgemm.cpp) — naive (메모리 바운드처럼 동작) → blocked (계산 바운드 쪽으로 이동).

---

## 4. 병렬 패턴 — GPU/CPU 공통 어휘

### 4-1. 5가지 기본 패턴
1. **Map** — `out[i] = f(in[i])`. trivially parallel. (예: elementwise)
2. **Reduce** — 모든 원소에서 결합 가능한 op로 1 값 산출. (예: sum)
3. **Scan (prefix sum)** — `out[i] = sum(in[0..i])`. Hillis-Steele / Blelloch.
4. **Gather/Scatter** — 간접 인덱스 읽기/쓰기. (예: embedding)
5. **Stencil** — 인접 원소 함수. (예: convolution)

**면접 질문**: "1부터 N까지 prefix sum을 GPU에서 어떻게?" — Hillis-Steele는 O(N log N) work, Blelloch는 O(N) work + O(log N) depth. Blelloch이 일반적으로 정답.

### 4-2. CUDA 워프 패턴
- `__shfl_sync` — 워프 안에서 레지스터 데이터를 직접 옮김 (shared mem 안 거침).
- `__ballot_sync`, `__activemask` — 워프 마스크 연산.
- 워프 트리 reduce: 5단계 `__shfl_xor_sync`로 32→1.

### 📌 데모
- [`../demos/02_reduce.cpp`](../demos/02_reduce.cpp) — 계층적 reduce.
- [`../demos/04_transpose.cpp`](../demos/04_transpose.cpp) — gather/scatter의 캐시 친화 변형.

---

## 5. CUDA 사고법 — 모르면 답이 좁아진다

CUDA 자체를 묻지는 않더라도 *CUDA를 아는 사람의 답이 더 깊다*. 최소한:

- **SIMT** — 32 thread (워프)가 같은 PC를 공유. 분기 시 직렬화 (warp divergence).
- **Occupancy** — SM당 active warp 비율. 너무 많은 레지스터/shared mem 쓰면 occupancy ↓.
- **Coalescing** — 한 워프의 32 thread가 연속 주소를 읽으면 1-2 트랜잭션, strided면 32 트랜잭션.
- **Bank conflict** — shared mem은 32 banks (4B 단위). 같은 워프가 같은 bank의 다른 word를 동시 접근하면 직렬화.
- **Tensor Core (MMA)** — `mma.m16n8k16` 같은 PTX. FP16/BF16/FP8 입력 → FP32 누산.
- **Async copy** — Ampere `cp.async`로 글로벌→shared 복사 + ldgsts로 비동기 파이프라이닝.
- **Hopper**: TMA (Tensor Memory Accelerator), WGMMA (warp group MMA), thread block clusters.

이 폴더의 자매 폴더 [`../kernels/`](../kernels/) 에서 19개 커널을 한국어 라인-바이-라인으로 해설합니다. 면접 전날 **08-sgemm**, **10-hgemm**, **11-flash-attn**, **12-swizzle** 4개는 반드시 다시 읽기를 권합니다.

---

## 6. 자주 나오는 코딩 라이브 문제 (NVIDIA 스타일)

| 문제 | 핵심 시그널 | 함정 |
|------|-----------|------|
| 두 정렬된 배열 합치기 | move semantics, 반복자 | 임시 컨테이너 복사 |
| Lock-free 단일 생산자/소비자 큐 | release-acquire | seq_cst 남용 |
| `std::function` 같은 type-erasure 컨테이너 | 작은 객체 최적화, vptr | 가상 함수 호출 비용 |
| 1024-bit 정수 +/* | 캐리 처리, intrinsics, SIMD | naive `for` |
| 큰 행렬의 합 (cache-friendly) | row vs column traversal | strided access |
| Producer-consumer ring buffer | atomic head/tail, ABA | 정수 오버플로 |
| 문자열에서 숫자 파싱 | branch prediction, lookup | regex 사용 |
| 4096개 객체 풀 | placement new, alignment | 계산 누락 |

각 문제마다 **"왜 이 답이 빠른가?"** 를 데이터 (cache miss, branch, instruction count) 로 설명할 수 있어야 합니다.

---

## 7. 한 페이지 면접 직전 체크리스트

- [ ] move ctor에 `noexcept` 빼먹지 않기
- [ ] `auto& x : container` 로 *복사* 안 만들기
- [ ] reduce/scan은 *계층적*, 공용 atomic 안 씀
- [ ] shared atomic 자리에 thread-local accumulator + 마지막 합
- [ ] cache line 64B (Apple Silicon은 128B) — false sharing 피하기
- [ ] release-acquire가 lock-free의 80%를 해결한다
- [ ] GEMM이 compute-bound가 *되게 만드는* 게 타일링이다
- [ ] FlashAttention = SRAM 안에서 online softmax
- [ ] CUDA 워프 = 32 thread + 같은 PC
- [ ] `if constexpr` / SFINAE / concepts 차이 한 줄로 설명 가능

---

## 8. 다음 단계

1. **지금 당장**: [`../demos/`](../demos/) 의 `make check` → `make run`으로 8개 데모 모두 한 번 돌려 보기.
2. **이번 주**: 데모 1개당 출력의 *해석*을 자기 말로 적어 보기 (한 단락).
3. **다음 주**: [`../kernels/`](../kernels/) 에서 본인이 약한 영역 (SGEMM/Flash Attention/Swizzle) 정독.
4. **면접 직전**: 본 페이지 § 7 체크리스트 + 데모 8개의 "포인트" 절을 30초씩 자기 말로.

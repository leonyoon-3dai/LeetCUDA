# 🧪 Mac에서 돌아가는 C++ 데모 — NVIDIA 면접 대비

> NVIDIA GPU가 없는 Mac (Intel/Apple Silicon 모두) 에서 **CUDA의 핵심 개념을 CPU C++로 똑같이 재현**한 데모 모음입니다. 각 데모는 단일 `.cpp` 파일이고, 200줄 이하, 한국어 주석으로 면접에서 바로 인용할 수 있는 포인트를 정리해 두었습니다.

이 폴더의 목적은 두 가지:

1. **C++ 역량을 측정 가능한 방식으로 굳히기** — "안다"가 아니라 "내 손으로 빌드해서 숫자를 봤다."
2. **CUDA 개념을 CPU 등가물로 매핑** — GPU가 없어도 *왜 빠른지*를 체감할 수 있게.

---

## 0. 환경 점검 (가장 먼저!)

이 Mac (또는 사용자 환경)이 빌드 가능한지 한 줄로 확인:

```bash
cd docs-kr/demos
make check
```

기대 출력 예:
```
CXX      = clang++
Apple clang version 15.x.x ...
ARCH     = x86_64
OS       = Darwin 22.6.0
AVX      : AVX1.0
AVX2     : no
Cores    : 4
```

### ❗ Xcode Command Line Tools가 망가져 있다면

아래와 같은 에러가 나오면:
```
xcrun: error: invalid active developer path (...), missing xcrun at: ...
```

다음 두 줄로 해결합니다 (인터랙티브이므로 사용자가 직접 실행해 주세요. Claude Code 프롬프트에서 `! xcode-select --install` 식으로):

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install   # GUI 팝업이 뜸 — '설치' 클릭
```

설치(약 5–10분) 후 다시 `make check`.

> **Apple Silicon (M1/M2/M3)** 사용자: 위와 동일하게 동작. `-march=native`가 자동으로 ARMv8 NEON으로 잡힙니다 (AVX는 비활성, 다른 SIMD 경로로 컴파일됨).

---

## 1. 데모 일람

| # | 파일 | CUDA 개념과의 매핑 | 면접 포인트 |
|---|------|-------------------|-------------|
| 01 | [`01_elementwise.cpp`](./01_elementwise.cpp) | `float4` 벡터화 로드 | 자동 벡터화의 조건 (restrict, aliasing), AVX intrinsics |
| 02 | [`02_reduce.cpp`](./02_reduce.cpp) | 워프 셔플 트리 reduce | 계층적 reduce vs 공용 atomic 안티패턴 |
| 03 | [`03_softmax_online.cpp`](./03_softmax_online.cpp) | Online softmax (FlashAttention) | (m, l) 병합 법칙 — 1-패스 softmax |
| 04 | [`04_transpose.cpp`](./04_transpose.cpp) | shared memory 타일 전치 | 캐시 라인 / strided access / blocking |
| 05 | [`05_sgemm.cpp`](./05_sgemm.cpp) | SGEMM BM/BN/BK 타일링 | 산술 강도 / loop reorder / blocking |
| 06 | [`06_false_sharing.cpp`](./06_false_sharing.cpp) | bank conflict | MESI 핑퐁 / `alignas(64)` |
| 07 | [`07_memory_order.cpp`](./07_memory_order.cpp) | `__threadfence` | release-acquire / seq_cst / relaxed |
| 08 | [`08_move_rvo.cpp`](./08_move_rvo.cpp) | (순수 C++) | RVO / NRVO / move / forward / Rule of 5 |

### 한 번에 다 돌려 보기

```bash
make run        # 전체 빌드 후 8개 데모 순서대로 실행
```

### 개별로 돌리기

```bash
make 01 && ./01_elementwise           # 또는 `make run-01`
make 03 && ./03_softmax_online 65536  # N=65536으로
```

---

## 2. 각 데모가 답해야 할 면접 질문

### 01 — Elementwise / 벡터화
- **Q**: "왜 CUDA의 `float4` 로드가 더 빠른가요?"
- 데모로 보여주는 답: 한 번의 메모리 트랜잭션에 더 많은 바이트를 묶어 트랜잭션 오버헤드를 줄임. CPU AVX도 동일 — 32B/명령어. `restrict` 없으면 컴파일러가 별칭 가능성을 의심해 보수적으로 컴파일하므로 "면접 코드에 `restrict`를 안 넣은 이유"라는 함정 질문도 종종 나옵니다.

### 02 — Reduce
- **Q**: "10⁹개 원소의 합을 GPU에서 어떻게 구현하시겠습니까?"
- 안티패턴 vs 정답: "공용 atomic에 더하지 마세요." 워프 안 → 블록 안 → 그리드 합산. CPU에서도 같은 패턴 (per-thread accumulator + 마지막 합산).

### 03 — Online softmax
- **Q**: "FlashAttention이 왜 메모리 효율적인가요?"
- 핵심: attention 행렬 `S = QK^T`를 풀로 메모리에 만들지 않고, K를 청크 단위로 SRAM(공유 메모리)에 올려 **online softmax**로 누적. 데모는 (m, l) 상태로 청크를 합치는 법칙이 정확히 일치함을 보입니다.

### 04 — Transpose
- **Q**: "캐시는 왜 transpose에서 망가지나요? CUDA는 어떻게 푸나요?"
- CPU: row-major × stride-N → 캐시라인 미스. Blocking으로 두 방향 재사용.
- CUDA: `__shared__ float tile[32][33]` (33은 bank conflict 회피용 패딩) → 합체 쓰기.

### 05 — SGEMM
- **Q**: "GEMM은 compute-bound인가, memory-bound인가? 왜?"
- 답: 충분한 산술 강도(arithmetic intensity)를 *얻으면* compute-bound. 그걸 가능케 하는 게 타일링 + 레지스터 블로킹. 데모로 naive vs blocked 차이를 직접 측정.

### 06 — False sharing
- **Q**: "두 스레드가 같은 캐시 라인의 다른 변수를 갱신하면 어떻게 되나요?"
- 답: MESI 핑퐁으로 사실상 직렬화. 해결: `alignas(std::hardware_destructive_interference_size)` 또는 64B 패딩.
- **연결**: CUDA bank conflict와 본질이 같다 — "같은 물리 자원을 동시에" → 직렬화.

### 07 — Memory order
- **Q**: "release/acquire와 seq_cst의 차이는요?"
- 답: 데모의 (1) 패턴이 핵심. release한 thread의 모든 이전 write를 acquire한 thread가 봄 (한 쌍에 한해서). seq_cst는 전역 순서까지 보장 — 비싸므로 진짜 필요할 때만.

### 08 — Move / RVO
- **Q**: "RVO와 std::move의 차이는요?"
- 답: RVO/NRVO는 컴파일러가 *반환 자리에 직접 객체를 짓는* 최적화. `std::move`는 lvalue → xvalue 캐스트일 뿐, 진짜 이동은 클래스의 move ctor가 합니다. C++17부터는 prvalue copy elision이 *guaranteed*.

---

## 3. 빌드 옵션 / 디버깅 팁

`Makefile`은 `-O3 -march=native -ffast-math -pthread`로 빌드합니다.

### 최적화 효과 직접 확인하기

```bash
make 05_sgemm                         # -O3 빌드
./05_sgemm 384 3                      # GFLOPS 출력

# -O0 비교 빌드 (Makefile의 패턴 룰)
make 05_sgemm_O0
./05_sgemm_O0 384 3                   # 보통 10배 이상 느림
```

### 어셈블리로 자동 벡터화 확인

```bash
clang++ -std=c++17 -O3 -march=native -S 01_elementwise.cpp -o 01.s
grep -E 'vmov|vadd|ymm|xmm' 01.s | head
# ymm 레지스터(256-bit AVX)가 보이면 자동 벡터화 성공
```

### Linux/Colab으로 실 GPU 테스트하기

이 폴더의 데모는 **CPU**용입니다. 실제 CUDA 커널 (이 저장소의 `kernels/*.cu`)을 돌려 보고 싶다면 무료 옵션:

- **Google Colab** — `Runtime → Change runtime type → T4 GPU` (무료 티어)
- **Kaggle Notebooks** — 무료 T4 ×2, 주당 30시간

```bash
# Colab 셀에서:
!nvidia-smi
!git clone https://github.com/xlite-dev/LeetCUDA && cd LeetCUDA && \
   pip install -e . && \
   python kernels/elementwise/elementwise.py
```

> 이 저장소의 상세 컴파일 절차는 루트 [`README.md`](../../README.md) 의 "Installation" 절을 참고.

---

## 4. 다음 단계

1. **데모 8개를 모두 돌리고** 출력 숫자를 외우지 말고 *해석*해 보기.
2. [`docs-kr/nvidia-cpp/`](../nvidia-cpp/) 의 면접 학습 로드맵 따라가기.
3. 특정 커널이 궁금하면 [`docs-kr/kernels/`](../kernels/) 에서 한국어 라인-바이-라인 해설 읽기.
4. 면접 직전: 각 데모의 "포인트" 절을 30초 안에 자기 말로 설명할 수 있는지 점검.

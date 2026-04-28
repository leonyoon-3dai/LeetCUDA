# LeetCUDA 한국어 데모 — Mac (Intel/Apple Silicon, NVIDIA GPU 불필요)
#
# 사용법:
#   make            # 전체 빌드
#   make run        # 전체 빌드 후 실행 (8개 데모 순서대로)
#   make 01         # 1번 데모만 빌드
#   make run-01     # 1번 데모 빌드+실행
#   make clean
#
# 요구 사항:
#   - clang++ 또는 g++ (C++17)
#   - macOS면 Xcode Command Line Tools 필요:
#       xcode-select --install
#     이미 설치돼 있는데 xcrun 에러가 나면:
#       sudo xcode-select --reset
#       sudo xcode-select -s /Library/Developer/CommandLineTools
#
# 이 Mac이 Intel i7-7660U (AVX 있음 / AVX2 없음) 같은 저사양이라도 동작합니다.
# AVX2가 필요한 경로는 자동으로 비활성됩니다 (#ifdef __AVX2__).

CXX      ?= clang++
CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-function
OPT      := -O3 -ffast-math
ARCH     := -march=native
THREADS  := -pthread

# pthread 링크 (macOS에서는 보통 없어도 되지만 안전하게)
LDFLAGS  := $(THREADS)
CXXFLAGS := $(CXXSTD) $(WARN) $(OPT) $(ARCH) $(THREADS)

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:.cpp=)

# 비최적화 빌드 (벤치마크 비교용)
DBG_BINS := $(addsuffix _O0,$(BINS))

.PHONY: all run clean help check

all: $(BINS)

# 개별 데모 단축어 (예: make 01 → 01_elementwise)
01: 01_elementwise
02: 02_reduce
03: 03_softmax_online
04: 04_transpose
05: 05_sgemm
06: 06_false_sharing
07: 07_memory_order
08: 08_move_rvo

run-01: 01_elementwise        ; @echo "── [01] elementwise ──"          ; ./01_elementwise
run-02: 02_reduce             ; @echo "── [02] reduce ──"               ; ./02_reduce
run-03: 03_softmax_online     ; @echo "── [03] online softmax ──"       ; ./03_softmax_online
run-04: 04_transpose          ; @echo "── [04] cache-blocked transpose ──"; ./04_transpose
run-05: 05_sgemm              ; @echo "── [05] tiled sgemm ──"          ; ./05_sgemm
run-06: 06_false_sharing      ; @echo "── [06] false sharing ──"        ; ./06_false_sharing
run-07: 07_memory_order       ; @echo "── [07] memory order ──"         ; ./07_memory_order
run-08: 08_move_rvo           ; @echo "── [08] move / RVO ──"           ; ./08_move_rvo

run: all
	@for b in $(BINS); do echo "── $$b ──"; ./$$b || exit $$?; echo; done

# 단일 .cpp → 실행 파일
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# -O0 비교 빌드 (최적화 끔; 벤치마크에서 최적화 효과 확인용)
%_O0: %.cpp
	$(CXX) $(CXXSTD) $(WARN) -O0 -g $(THREADS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS) $(DBG_BINS) *.dSYM
	rm -rf *.dSYM

# 환경 점검
check:
	@echo "CXX      = $(CXX)"
	@$(CXX) --version | head -1
	@echo "ARCH     = $(shell uname -m)"
	@echo "OS       = $(shell uname -s) $(shell uname -r)"
	@echo "AVX      : $(shell sysctl -n machdep.cpu.features 2>/dev/null | grep -o 'AVX1\.0' || echo 'no')"
	@echo "AVX2     : $(shell sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -o 'AVX2' || echo 'no')"
	@echo "Cores    : $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)"

help:
	@echo "make           — 모든 데모 빌드"
	@echo "make run       — 전체 빌드 후 실행"
	@echo "make 01..08    — 개별 데모 빌드"
	@echo "make run-01..08— 개별 데모 빌드+실행"
	@echo "make check     — 컴파일러/CPU 환경 점검"
	@echo "make clean     — 빌드 산출물 삭제"

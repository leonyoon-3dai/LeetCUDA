# 12 · Swizzle — XOR 기반 뱅크 충돌 회피

> 원본 파일:
> - [`kernels/swizzle/README.md`](../../kernels/swizzle/README.md)
> - [`kernels/swizzle/mma_simple_swizzle.cu`](../../kernels/swizzle/mma_simple_swizzle.cu)
> - [`kernels/swizzle/hgemm_mma_swizzle.cu`](../../kernels/swizzle/hgemm_mma_swizzle.cu)
> - [`kernels/swizzle/mat_trans_swizzle.cu`](../../kernels/swizzle/mat_trans_swizzle.cu)
>
> **핵심 학습 포인트**:
> 1. 패딩(+PAD) 방식의 **SMEM 용량 낭비 문제**.
> 2. **XOR 재주소화**로 같은 용량을 유지하며 100% 충돌 제거.
> 3. `ldmatrix` / MMA 접근 패턴에 swizzle이 왜 거의 필수인지.

---

## 1. 패딩 방식의 한계 복기

[07-transpose.md](./07-transpose.md)에서 배운 `+PAD=1` 기법은 SMEM 행에 1~8개 더미 원소를 끼워 **행 간 뱅크 오프셋**을 시프트했습니다. 이 방식은:

- ✅ 간단
- ❌ SMEM 용량 증가 (행마다 +N칸)
- ❌ 특정 MMA 레이아웃에선 **완전 해결 불가** (16×16 타일을 여러 개 배열 시 경계에서 재충돌)

큰 타일 (예: `128×128` HGEMM) 에서는 패딩만으로는 `ldmatrix`의 모든 접근 패턴을 커버하기 어려움.

---

## 2. Swizzle의 아이디어

**핵심**: SMEM의 **저장 위치 자체를 뒤섞어** 동일한 논리 주소가 다른 물리 뱅크에 매핑되게 함.

수식:
$$
\text{physical\_col} = \text{logical\_col} \oplus (\text{logical\_row} \bmod K)
$$

XOR 연산은 **가역적**이므로 같은 논리 주소끼리 충돌하지 않고, 서로 다른 논리 행은 **자동으로 다른 뱅크에 분산**됩니다.

### 가장 단순한 형태: `row ^ col`

```
logical_row:   col=0  col=1  col=2  col=3  col=4  col=5  col=6  col=7
   row=0:       B0    B1    B2    B3    B4    B5    B6    B7
   row=1:       B1    B0    B3    B2    B5    B4    B7    B6    ← row=1 전체가 XOR 1 시프트
   row=2:       B2    B3    B0    B1    B6    B7    B4    B5    ← row=2는 XOR 2 시프트
   row=3:       B3    B2    B1    B0    B7    B6    B5    B4
   ...
```

- **한 열 방향** (같은 col, 다른 row) 접근 시: 모두 **다른 뱅크** → 충돌 없음.
- **한 행 방향** (같은 row, 다른 col) 접근 시: 이미 원래도 각자 다른 뱅크.
- 용량 낭비 0. PAD 대비 우월.

### XOR의 bijection 성질

`(a, b)` 쌍에 대해 `c = a ^ b`는 `a`가 고정이면 `b`와 `c`가 1:1 대응. 그래서 같은 논리 주소를 두 번 접근하면 같은 물리 주소가 나오고, 데이터 손실 없음.

---

## 3. MMA와 swizzle

`ldmatrix.sync.aligned.m8n8.x4.shared.b16` 명령은 **32 스레드가 각자 16B씩** SMEM에서 읽어 레지스터에 적재합니다. 그 주소 패턴:

```
스레드   addr (byte offset)
  0      [row= 0][col= 0..7]   (fp16 × 8)
  1      [row= 1][col= 0..7]
  ...
  7      [row= 7][col= 0..7]
  8      [row= 0][col= 8..15]
  9      [row= 1][col= 8..15]
  ...
  15     [row= 7][col= 8..15]
  16     [row= 8][col= 0..7]
  ...
  31     [row=15][col= 8..15]
```

### 기본 저장으로 접근 시

`s_smem[row][col]` → 뱅크 = `(row * row_stride + col) / 16 % 32`.

32 스레드가 **row 0, 1, 2, ..., 7**을 주소당 16B씩 접근:
- row마다 col=0..7까지 동일 → 같은 뱅크에 여러 번 → **충돌**

### Swizzle 적용 시 (8×8 XOR)

행 r의 물리 col을 `c ^ (r & 0x7)` 로 재배치:
```
r=0:  col 0,1,2,3,4,5,6,7  → 뱅크 0..7
r=1:  col 1,0,3,2,5,4,7,6  → 뱅크 1,0,3,2,5,4,7,6   ← 동일 col=0이 뱅크 1로
r=2:  col 2,3,0,1,6,7,4,5  → ...
...
r=7:  ...
```

8개 스레드가 (row=0..7, col=0) 읽을 때 → 물리적으로 뱅크 0, 1, 2, 3, 4, 5, 6, 7 모두 다름. ✅

---

## 4. 원본 파일의 결과 (인용)

`kernels/swizzle/README.md`에서 NCU 측정치:

```
naïve hgemm_mma_m16n8k16:
  bank_conflicts_ldsm.sum = 2,097,152   (수백만 건)

hgemm_mma_m16n8k16_swizzle:
  bank_conflicts_ldsm.sum = 0           ★ 완전 제거
```

더 큰 `mma2x4_warp4x4` 커널에서도:
```
기본 : sum = 2,359,296
swizzle: sum = 0
```

**완전한 0**을 달성. 패딩으로는 쉽게 안 나오는 수치.

---

## 5. 실제 구현 — 간략 예시

(실제 LeetCUDA 코드는 템플릿과 매크로가 복잡해 요약 형태로 설명.)

### 저장 시

```cuda
// 논리 좌표 (row, col)에 값 v를 저장
int swizzled_col = col ^ (row & MASK);   // MASK=0x7 for 8-row swizzle
s_smem[row][swizzled_col] = v;
```

### 로드 시

```cuda
int swizzled_col = col ^ (row & MASK);
half v = s_smem[row][swizzled_col];
```

**저장/로드 모두에 같은 XOR 적용**. 그러면 원래 (row, col)에 저장했던 값이 동일 (row, col)로 읽힘.

### 마스크 선택

- `MASK = 0x7` (3비트): 8-way swizzle. 일반적.
- `MASK = 0x1F` (5비트): 32-way swizzle. 더 공격적.
- `MASK = 0` : swizzle 없음.

타일 크기와 MMA 패턴에 따라 최적값이 다름. 실험적으로 결정.

---

## 6. `mat_trans_swizzle.cu` — 전치에 적용

[07-transpose.md](./07-transpose.md)의 전치 커널에서:

**PAD 방식** (현행):
```cuda
__shared__ float tile[16][64 + 1];   // 256 B 추가
```

**Swizzle 방식**:
```cuda
__shared__ float tile[16][64];        // 낭비 없음
// 접근 시:
tile[row][col ^ (row & 0xF)] = ...;
```

4KB SMEM 아끼면 **occupancy 1단계 상승** 가능 (SM당 블록 수가 SMEM 한계로 결정되는 경우).

---

## 7. CUTLASS / CuTe 스타일

NVIDIA 공식 라이브러리 CUTLASS는 swizzle을 **레이아웃 타입**으로 추상화:

```cpp
using Layout = ComposedLayout<
    Swizzle<3, 3, 3>,  // <B, M, S>: MxB 블록 단위 S-bit swizzle
    Layout<...>>;
```

- `B=3`: 뱅크 영역 단위 log2 (8)
- `M=3`: 타일 단위 log2 (8)
- `S=3`: 시프트 단위 log2

LeetCUDA의 수동 구현은 이 원리를 직접 코드에 녹인 것. 학습 목적으로 매우 유익.

---

## 8. 언제 swizzle을 쓸까

| 상황 | 권장 |
|------|------|
| 작은 SMEM, 간단한 패턴 | **PAD** (이해 쉬움) |
| MMA `ldmatrix` 사용 | **Swizzle 거의 필수** |
| 대형 타일 (64+ 원소/행) | **Swizzle** |
| SMEM 용량이 빡빡 | **Swizzle** |
| prototyping | PAD, 튜닝 단계에서 swizzle로 전환 |

---

## 9. 시각 요약

```
────────────────── 기본 SMEM ──────────────────
col: 0  1  2  3  4  5  6  7  ...  31
r=0: B0 B1 B2 B3 B4 B5 B6 B7  ...  B31
r=1: B0 B1 B2 B3 B4 B5 B6 B7  ...  B31   ← 같은 col이 같은 뱅크
r=2: B0 B1 B2 B3 B4 B5 B6 B7  ...  B31
...
→ 열 접근 시 32-way conflict

──────────────── PAD (+1) SMEM ────────────────
col: 0  1  2  3  4  5  6  7  ...  31 32
r=0: B0 B1 B2 B3 B4 B5 B6 B7  ...  B31 B0
r=1: B1 B2 B3 B4 B5 B6 B7 B8  ...  B0  B1  ← 1 뱅크 시프트
r=2: B2 B3 ...
→ 열 접근 시 충돌 없음, 단 SMEM +N만큼 낭비

──────────────── Swizzle SMEM ────────────────
col: 0  1  2  3  4  5  6  7
r=0: B0 B1 B2 B3 B4 B5 B6 B7
r=1: B1 B0 B3 B2 B5 B4 B7 B6     ← col XOR 1
r=2: B2 B3 B0 B1 B6 B7 B4 B5     ← col XOR 2
r=3: B3 B2 B1 B0 B7 B6 B5 B4     ← col XOR 3
...
→ 열 접근 시 충돌 없음, SMEM 낭비 0
```

---

## 10. 마치며 — 한국어 해설 시리즈 정리

이 `docs-kr` 시리즈는 CUDA 입문자가 다음 흐름을 따라 성장하도록 설계했습니다:

```
메모리 대역폭 (01, 02)  ───▶  워프 통신 (03, 04)
                              ▼
수치 알고리즘 (05, 06)  ◀───  타일링 기초 (07)
        ▼
범용 GEMM (08, 09)  ───▶  Tensor Core (10)
                              ▼
                        Flash Attention (11) ──▶ Swizzle (12)
```

각 문서가 **앞선 개념의 조합**으로 이어지므로, 앞에서부터 순차 학습을 강력 권장합니다.

추가로 본 번역의 대상이 되지 못한 커널들 (embedding, nms, rope, gelu, elu, hardswish, sgemv, hgemv, transformer 등)은 **원 저자의 주석**과 본 시리즈의 `01~04` 문서가 다룬 패턴을 조합한 응용입니다. 관심 있으면 `kernels/` 디렉터리에서 직접 읽어 보시길.

---

## 원저작권 / 면책

본 해설 문서군은 [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)의 학습 목적 번역/주석 레이어입니다. 원 코드와 라이선스(GPL v3)는 원 저작자에게 있습니다. 본 문서의 오역이나 오해가 있으면 원 저자의 코드와 의도가 정확합니다.

🙏 원 저자 [@DefTruth](https://github.com/DefTruth) 및 기여자들께 감사.

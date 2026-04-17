# 18 · WS-HGEMM — Warp Specialization (Hopper)

> 원본 파일:
> - [`kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu`](../../kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu) (CUTLASS CuTe 기반)
> - [`kernels/hgemm/wgmma/`](../../kernels/hgemm/wgmma/) (WGMMA PTX)
>
> **핵심 학습 포인트**:
> 1. 블록 내 워프의 **역할 분리**: producer (로드 전담) vs consumer (MMA 전담).
> 2. **Mbarrier** 동기화 — `__syncthreads()`를 넘어선 세밀 제어.
> 3. Hopper의 **TMA + WGMMA** 조합이 왜 현재 최고 성능인가.

---

## 1. 왜 Warp Specialization?

[09-sgemm-async](./09-sgemm-async.md)의 이중 버퍼는 **모든 워프가 같은 코드**를 실행했습니다:

```
각 워프:  cp.async 발사 → mma 실행 → cp.async 발사 → mma 실행 → ...
                               (주기적 반복)
```

문제: MMA는 Tensor Core를 많이 쓰고, `cp.async`는 메모리 하위시스템을 많이 씁니다. **역할이 달라서 스케줄러가 혼란스러워하는 구간**이 생김.

**Warp Specialization**: 워프마다 역할을 나눔. 일부는 계속 로드만, 일부는 계속 계산만.

```
Producer 워프들: 계속 cp.async 발사, 다음 타일을 준비
Consumer 워프들: 준비된 타일로 MMA만 계속 수행

동기화: mbarrier로 producer "다 로드됨" → consumer "사용 가능"
```

### 전통 vs Warp Spec 시각화

```
전통 파이프라인 (모든 워프 동일):
  W0: Ld━━━━Co━━━━Ld━━━━Co━━━━Ld━━━━Co━━━━
  W1: Ld━━━━Co━━━━Ld━━━━Co━━━━...
  → 각 워프가 두 역할 전환, 레지스터/ICache 압력 ↑

Warp Specialized (Producer 전담 / Consumer 전담):
  W0 (P): Ld━━Ld━━Ld━━Ld━━Ld━━Ld━━Ld━━Ld━━  (메모리 파이프만 사용)
  W1 (C): Co━━━━Co━━━━Co━━━━Co━━━━Co━━━━  (TC만 사용)
  W2 (C): Co━━━━Co━━━━...
  W3 (C): Co━━━━...
  → 하드웨어 유닛이 명확히 분리, 더 높은 throughput
```

---

## 2. Hopper의 새 요소들

### TMA (Tensor Memory Accelerator)

- **1 스레드가 발사** → 하드웨어가 자동으로 2D/3D 타일을 DRAM → SMEM 복사.
- 주소 계산, 워프 협력 로드, 경계 처리 모두 **HW 자동화**.
- 완료 시 mbarrier 갱신.

```cuda
// 의사 PTX
cp.async.bulk.tensor.2d.shared.global [smem_ptr], [tensor_map, coord], [mbar];
// → 1 명령으로 한 타일 전체 복사 시작
```

### WGMMA (Warp-Group MMA)

- **128 스레드(4 워프 = warp group)** 가 협력.
- `wgmma.mma_async m64nNk16` 등의 shape.
- **비동기**: 발사 후 다른 일 하다 `wgmma.wait_group` 으로 완료 확인.

```cuda
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

### Mbarrier

- SMEM 공간에 저장되는 **도착 카운터**.
- Producer가 `mbarrier.arrive` → Consumer가 `mbarrier.wait_parity`.
- `__syncthreads()` 보다 훨씬 **유연한 제한된 동기화**.

---

## 3. 본 파일이 SM8x (Ampere) 버전인 이유

`naive_ws_hgemm_sm8x.cu` 는 **Ampere에서도 돌도록** WS 패턴을 구현한 예시. Hopper 고유 명령(TMA, WGMMA) 없이 다음으로 근사:

- TMA 대신 `cp.async` (Ampere도 지원)
- WGMMA 대신 `mma.m16n8k16` 반복
- mbarrier 대신 `cuda::barrier` 또는 cooperative groups 파이프라인

CUTLASS의 **CuTe DSL**을 활용:

```cpp
using mma_op    = SM80_16x8x16_F16F16F16F16_TN;   // Ampere MMA op
using mma_atom  = MMA_Atom<MMA_Traits<mma_op>>;
using TiledMMA  = decltype(make_tiled_mma(mma_atom{}, ThrLayout, Permute));
```

`TiledMMA`는 **여러 MMA atom을 타일로 묶어** CuTe가 자동으로 스레드 매핑을 처리. 수동 PTX 대비 훨씬 적은 코드로 같은 성능.

---

## 4. Producer/Consumer 코드 구조 (개념)

```cpp
if (warp_id < kProducerWarps) {
    // ─── Producer ───
    for (int tile = tile_id_start; tile < num_tiles; tile += producer_stride) {
        // 다음 스테이지의 mbarrier가 "consumer 작업 완료" 상태인지 확인
        mbarrier.wait_parity(...);

        // cp.async.bulk (TMA) 또는 cp.async (Ampere) 로 타일 로드
        cp.async.bulk(&smem[stage], &gmem[tile], &mbarrier);
    }
} else {
    // ─── Consumer ───
    TiledMMA tmma;
    auto acc = partition_fragment_C(tmma, Shape<BM, BN>{});
    clear(acc);

    for (int tile = 0; tile < num_tiles; ++tile) {
        int stage = tile % kStages;
        // Producer 로드 완료 대기
        mbarrier[stage].wait_parity(...);

        // ldmatrix (A, B) → MMA
        auto thr_A = tmma.partition_fragment_A(smem_A[stage]);
        auto thr_B = tmma.partition_fragment_B(smem_B[stage]);
        gemm(tmma, acc, thr_A, thr_B, acc);

        // 이 stage 완료 알림 (producer가 다음 타일 덮어써도 됨)
        mbarrier[stage].arrive();
    }

    // 누산된 C를 SMEM/DRAM으로
    store(acc, C);
}
```

### 핵심 파라미터

- `kStages = 3~4`: SMEM 버퍼 개수. 많을수록 오버랩 ↑, SMEM ↓.
- `kProducerThread`: 로드 전담 스레드 수 (보통 32 or 64).
- `kConsumerThread`: MMA 참여 스레드 (128 or 256).

---

## 5. Swizzle 통합

[12-swizzle](./12-swizzle.md)에서 배운 XOR swizzle이 CuTe에서는 레이아웃 타입으로 추상화:

```cpp
using SmemLayoutAtom =
    decltype(composition(
        Swizzle<3, 3, 3>{},         // <B=3, M=3, S=3>
        make_layout(Shape<_8, _kCTAK>{}, LayoutRight{})
    ));
```

`Swizzle<3,3,3>` = 8×8 블록 단위 8-bit 시프트 swizzle. 컴파일 타임에 주소 변환이 코드에 녹아들어 **런타임 오버헤드 0**.

---

## 6. 성능 (원 README 인용)

원 저자 측정 (N=4096, 정확한 장비 불명 — H100 추정):

```
ws_hgemm_naive_cute:  3.72 ms
torch fp16 (cuBLAS):  5.06 ms
```

이 간단한 **WS + CuTe** 구현만으로도 **cuBLAS보다 36% 빠름** (특정 shape/크기 한정). 프로덕션 최적화를 더하면 더 격차 커질 수 있음.

---

## 7. 실무 관점

**공부 목적**: Hopper 프로그래밍 모델 이해.

**프로덕션 사용**:
- 이 수준의 커널을 **직접 작성**하기보다 **CUTLASS 또는 cuBLAS-Lt**의 kernel 선택 기능 사용.
- 특수 shape(배치/작은 N)나 퓨전이 필요한 경우에만 커스텀.

**학습 경로 권장**:
1. CuTe 튜토리얼 (NVIDIA 공식)
2. CUTLASS 예제 중 `48_hopper_warp_specialized_gemm`
3. FlashAttention-3 소스 (TMA + WGMMA + WS 종합)

---

## 8. 마치며

WS-HGEMM은 **본 시리즈의 모든 테크닉**이 집약된 현재 최첨단:
- 타일링 ([08](./08-sgemm.md))
- 비동기 복사 ([09](./09-sgemm-async.md))
- Tensor Core ([10](./10-hgemm.md))
- Swizzle ([12](./12-swizzle.md))
- + Warp Specialization (본 문서)

+ 하드웨어 기능 (TMA, WGMMA, mbarrier).

이 위에 Flash Attention의 아이디어([11](./11-flash-attn.md))가 얹혀 **FlashAttention-3**가 됩니다.

---

## 다음 문서

👉 [19-hardswish-hardshrink.md](./19-hardswish-hardshrink.md) — 다시 기초로 돌아가 빠트린 활성화 함수들을 보강.
